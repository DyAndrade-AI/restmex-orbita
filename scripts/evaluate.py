#!/usr/bin/env python3
"""Evaluación automática: un comando genera el reporte completo para quienes evalúan.

    python scripts/evaluate.py              # léxico (igual que Docker): gold, baselines, robustez
    python scripts/evaluate.py --holdout    # además, benchmark autosupervisado sin fuga
    python scripts/evaluate.py --semantic   # híbrido con E5 (carga torch: puede tardar minutos)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender import Recommender  # noqa: E402
from restmex_recommender import evaluation as ev  # noqa: E402

GOLD = ROOT / "eval/gold_queries.json"
# Compatibilidad con scripts/evaluate_futures.py.
QUERIES = [item["query"] for item in ev.load_gold(GOLD)[:10]]


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown(report: dict) -> str:
    gold = report["gold"]["aggregate"]
    lines = [
        "# Reporte de evaluación automática",
        "",
        f"Generado: {report['generated_at']} · modo **{report['mode']}** · "
        f"duración {report['duration_s']} s",
        "",
        "> “5 mejores” = mayor afinidad temática con la consulta. “5 peores” = menor afinidad; "
        "no implica peor calidad. La calidad (polaridad con prior bayesiano) se reporta aparte.",
        "",
        "## Rúbrica",
        "",
        "| Criterio | Valor | Umbral | Estado |",
        "|---|---|---|---|",
    ]
    for row in report["rubric"]:
        state = "✅" if row["pass"] else "❌"
        lines.append(f"| {row['check']} | {_fmt(row['value'])} | {row['threshold']} | {state} |")
    lines += [
        "",
        "## Gold seed (heurístico, pendiente de adjudicación humana)",
        "",
        "| Métrica | Sistema | Popularidad | Azar |",
        "|---|---|---|---|",
    ]
    base = report["baselines"]
    for key in ["precision_at_5", "capped_recall_at_5", "hit_at_1", "hit_at_5", "mrr",
                "ndcg_at_5", "bottom_error_rate", "exclusion_violation_rate"]:
        lines.append(
            f"| {key} | {_fmt(gold[key])} | {_fmt(base['popularity'][key])} | "
            f"{_fmt(base['random'][key])} |"
        )
    lines += ["", "### Ablación de señales", "", "| Configuración | Hit@1 | MRR | nDCG@5 |",
              "|---|---|---|---|"]
    for name, agg in report["ablation"].items():
        lines.append(f"| {name} | {_fmt(agg['hit_at_1'])} | {_fmt(agg['mrr'])} | "
                     f"{_fmt(agg['ndcg_at_5'])} |")
    if report.get("holdout"):
        h = report["holdout"]
        m, p = h["metrics"], h["popularity_baseline"]
        lines += [
            "",
            "## Holdout autosupervisado (sin consultas escritas a mano)",
            "",
            h["description"],
            "",
            f"{h['evaluated_queries']} reseñas reservadas · modo {h['mode']}",
            "",
            "| Métrica | Sistema | Popularidad | Azar |",
            "|---|---|---|---|",
            f"| Hit@1 | {_fmt(m['hit_at_1'])} | {_fmt(p['hit_at_1'])} | {1 / 40:.3f} |",
            f"| Hit@5 | {_fmt(m['hit_at_5'])} | {_fmt(p['hit_at_5'])} | "
            f"{h['random_expectation']['hit_at_5']} |",
            f"| MRR | {_fmt(m['mrr'])} | {_fmt(p['mrr'])} | — |",
            f"| Pueblo correcto en bottom 5 | {_fmt(m['bottom_error_rate'])} | "
            f"{_fmt(p['bottom_error_rate'])} | {h['random_expectation']['bottom_error_rate']} |",
            "",
            f"Spearman(Hit@5 por pueblo, nº de reseñas) = {h['spearman_hit5_vs_reviews']}",
        ]
    bias = report["volume_bias"]
    lines += [
        "",
        "## Sesgo por volumen de reseñas",
        "",
        f"Spearman(apariciones en top 5, nº de reseñas) = {bias['spearman_exposure_vs_reviews']} "
        f"(p = {bias['p_value']}). Pueblos que nunca aparecen en el top 5 del gold: "
        f"{', '.join(bias['towns_never_in_top5']) or 'ninguno'}.",
        "",
        "## Robustez y casos adversariales",
        "",
        f"Jaccard medio del top 5 ante variantes (sin acentos, relleno, orden): "
        f"{report['robustness']['mean_top5_jaccard']} (mínimo "
        f"{report['robustness']['min_top5_jaccard']}).",
        "",
        "| Caso | Esperado | Resultado | OK |",
        "|---|---|---|---|",
    ]
    for case in report["adversarial"]["cases"]:
        lines.append(f"| {case['case']} | {case['expected']} | {case['outcome'][:80]} | "
                     f"{'✅' if case['pass'] else '❌'} |")
    lines += ["", "## Consultas con fallos (Hit@5 = 0 o relevante en bottom)", ""]
    failures = [q for q in report["gold"]["queries"]
                if q["hit_at_5"] == 0 or q["relevant_in_bottom"] > 0]
    for q in failures:
        lines.append(f"- **{q['id']}** “{q['query']}” → top {', '.join(q['top'])}; "
                     f"esperados {', '.join(q['relevant'])}")
    if not failures:
        lines.append("- Ninguna.")
    human = report["human"]
    lines += ["", "## Evaluación humana ciega", ""]
    if human.get("sessions"):
        lines.append(
            f"{human['sessions']} sesiones de {human['evaluators']} evaluadores · "
            f"precision@5 humana {human['human_precision_at_5']} · bottom no relacionado "
            f"{human['human_bottom_unrelated_rate']} · acuerdo {human['pairwise_agreement']} · "
            f"mediana {human['median_seconds_per_query']} s por consulta"
        )
    else:
        lines.append("Sin sesiones todavía: usa la pestaña Evaluación de la web.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--semantic", action="store_true", help="Evalúa el modo híbrido con E5")
    parser.add_argument("--per-town", type=int, default=15)
    args = parser.parse_args()
    if args.per_town < 1:
        parser.error("--per-town debe ser al menos 1")

    started = time.perf_counter()
    model_dir = ROOT / "models/multilingual-e5-base"
    engine = Recommender(ROOT / "artifacts", model_dir, enable_semantic=args.semantic)
    print(f"Modo: {'híbrido' if engine.semantic_enabled else 'léxico'} · {engine.semantic_status}")
    if args.semantic and not engine.semantic_enabled:
        sys.exit("Se pidió --semantic pero el modelo no cargó; no se reporta como híbrido.")
    gold = ev.load_gold(GOLD)
    gold_result = ev.gold_suite(engine, gold)
    determinism = ev.determinism_suite(engine, gold)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "híbrido" if engine.semantic_enabled else "léxico",
        "semantic_status": engine.semantic_status,
        "gold_provenance": json.loads(GOLD.read_text(encoding="utf-8"))["provenance"],
        "rubric": ev.rubric(gold_result, determinism),
        "gold": gold_result,
        "baselines": ev.baseline_suite(engine, gold),
        "ablation": ev.ablation_suite(engine, gold),
        "robustness": ev.robustness_suite(engine, gold),
        "adversarial": ev.adversarial_suite(engine),
        "determinism": determinism,
        "volume_bias": ev.volume_bias_suite(engine, gold_result["queries"]),
        "human": ev.human_label_suite(ROOT / "eval/human_labels.jsonl"),
    }
    if args.holdout:
        report["holdout"] = ev.holdout_suite(
            ROOT / "data/Rest-Mex_2025_train_clean.csv",
            ROOT / "artifacts/holdout",
            per_town=args.per_town,
            with_semantic=engine.semantic_enabled,
            model_dir=model_dir if engine.semantic_enabled else None,
        )
    report["duration_s"] = round(time.perf_counter() - started, 1)
    (ROOT / "artifacts/evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = markdown(report)
    (ROOT / "docs/evaluation_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
