#!/usr/bin/env python3
"""Ejecuta la comparación TF-IDF / evidencia / BM25 / E5 / híbridos sobre el holdout.

    python scripts/experiments.py              # híbrido con E5 si carga
    python scripts/experiments.py --lexical    # solo señales léxicas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender import evaluation as ev  # noqa: E402
from restmex_recommender import experiments as ex  # noqa: E402


def table(report: dict) -> str:
    lines = ["| Sistema | Pesos | MRR (prueba) | Hit@1 | Hit@5 | nDCG@5 | En bottom 5 |",
             "|---|---|---|---|---|---|---|"]
    for name, row in report["systems"].items():
        t = row["test"]
        weights = ", ".join(f"{k}={v:.2f}" for k, v in row["weights"].items() if v)
        lines.append(f"| {name} | {weights} | {t['mrr']:.3f} | {t['hit_at_1']:.3f} | "
                     f"{t['hit_at_5']:.3f} | {t['ndcg_at_5']:.3f} | {t['bottom5_rate']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical", action="store_true")
    parser.add_argument("--per-town", type=int, default=15)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    started = time.perf_counter()
    model_dir = ROOT / "models/multilingual-e5-base"
    engine, _, queries = ev.prepare_holdout(
        ROOT / "data/Rest-Mex_2025_train_clean.csv",
        ROOT / ("artifacts/holdout" if args.lexical else "artifacts/holdout_semantic"),
        per_town=args.per_town,
        with_semantic=not args.lexical,
        model_dir=None if args.lexical else model_dir,
    )
    report = ex.run(engine, queries, step=args.step)
    report["duration_s"] = round(time.perf_counter() - started, 1)
    suffix = "lexical" if args.lexical else "hybrid"
    (ROOT / f"artifacts/experiments_{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    boot = report["tuned_vs_reference_test"]
    text = "\n".join([
        f"# Experimentos de recuperación ({suffix})", "",
        f"Holdout: {report['queries']['calibration']} consultas de calibración, "
        f"{report['queries']['test']} de prueba. Rejilla: {report['grid_size']} combinaciones "
        f"(paso {args.step}). Duración {report['duration_s']} s.", "",
        table(report), "",
        f"Ajustado vs {report['reference']} en prueba: ΔMRR = {boot['mrr']['mean_diff']:+.4f} "
        f"(IC95 {boot['mrr']['ci95_low']:+.4f} a {boot['mrr']['ci95_high']:+.4f}); "
        f"ΔHit@5 = {boot['hit_at_5']['mean_diff']:+.4f} "
        f"(IC95 {boot['hit_at_5']['ci95_low']:+.4f} a {boot['hit_at_5']['ci95_high']:+.4f}).",
        "",
        "Léxico en producción (con cobertura de facetas) vs léxico anterior en prueba: " + "; ".join(
            f"Δ{k.upper()} = {b['mean_diff']:+.4f} (IC95 {b['ci95_low']:+.4f} a {b['ci95_high']:+.4f})"
            for k, b in report["lexical_new_vs_previous_test"].items()) + ".",
    ]) + "\n"
    (ROOT / f"docs/experimentos_{suffix}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
