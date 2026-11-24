#!/usr/bin/env python3
"""Usuarios simulados para Strange Loop + sensibilidad de los pesos de rol de Órbita."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_multiverse import QUERIES as AUDIT_QUERIES  # noqa: E402

from restmex_recommender import Recommender  # noqa: E402
from restmex_recommender.futures import _extract_facets  # noqa: E402
from restmex_recommender.orbit_eval import (  # noqa: E402
    POPULATIONS, role_sensitivity, simulate_users, summarize_users,
)


OUT = ROOT / "artifacts/orbit_validation.json"
DOC = ROOT / "docs/orbit_validation.md"


def pct(value: float) -> str:
    return f"{100 * value:.1f} %"


def render_markdown(report: dict) -> str:
    """Tablas de docs/orbit_validation.md, generadas SOLO desde el artefacto (no a mano)."""
    sl = report["strange_loop"]
    lines = [
        "# Validación sintética de Strange Loop y Órbita",
        "",
        "> Generado por `scripts/validate_orbit.py` desde `artifacts/orbit_validation.json`. "
        "No editar a mano.",
        ">",
        "> **Tipo de evidencia: usuarios simulados.** No es evaluación humana: no hay juicios "
        "humanos en este experimento.",
        "",
        f"- Fecha: {report.get('generated_at', 'sin registrar')} · modo: {report['mode']} · "
        f"usuarios por consulta: {report.get('users_per_query', 'sin registrar')} · "
        f"semilla del usuario i = índice de la consulta · duración {report['duration_s']} s",
        f"- Consultas: {report.get('queries_total', 'sin registrar')} "
        "(banco + auditoría + ejemplo del video); las de una sola faceta no tienen preguntas y se omiten.",
        "",
        "| Población | Consultas | Ancla Hit@1 | SL 0 preg. | SL 1 preg. info / azar | "
        "SL 3 preg. info / azar | Hit@5 (3 preg.) | Mejora 3 vs 0 (IC95) | Info − azar, 1 preg. (IC95) | "
        "Azar > info (1 preg.) | Preguntar empeora (3) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in sl.items():
        info, rand, g1, g3 = row["informativa"], row["azar"], row["hit_at_1_gain"]["1"], row["hit_at_1_gain"]["3"]
        lines.append(
            f"| {name} | {row['queries']} | {pct(row['anchor_hit_at_1'])} | {pct(info['hit_at_1'][0])} | "
            f"{pct(info['hit_at_1'][1])} / {pct(rand['hit_at_1'][1])} | "
            f"{pct(info['hit_at_1'][3])} / {pct(rand['hit_at_1'][3])} | {pct(info['hit_at_5'][3])} | "
            f"{100 * g3['mejora_vs_sin_preguntas']:+.1f} pp [{100 * g3['ci95_mejora'][0]:.1f}, "
            f"{100 * g3['ci95_mejora'][1]:.1f}] | {100 * g1['informativa_menos_azar']:+.1f} pp "
            f"[{100 * g1['ci95'][0]:.1f}, {100 * g1['ci95'][1]:.1f}] | "
            f"{row.get('queries_where_random_beats_informative_at_1q', '—')} | "
            f"{row.get('queries_where_questions_hurt_at_3q', '—')} |")
    roles = report["role_sensitivity"]
    lines += [
        "",
        "Poblaciones: ver `POPULATIONS` en `src/restmex_recommender/orbit_eval.py`. *prior* usa la "
        "misma utilidad y el mismo Dirichlet que el modelo (caso optimista, coherencia interna); las "
        "demás rompen un supuesto a propósito. Hit@1 = el destino más frecuente del multiverso "
        "restante es el ideal oculto del usuario.",
        "",
        f"## Sensibilidad de los pesos de rol (±{int(100 * roles['spread'])} %, "
        f"{roles['draws_per_query']} sorteos por consulta, riesgo {roles['risk']})",
        "",
        "| Rol | Conserva el mismo destino |",
        "|---|---|",
        *[f"| {role} | {value} % |" for role, value in roles["role_unchanged_pct"].items()],
        "",
        f"Jaccard medio de la cartera: {roles['portfolio_jaccard_mean']} · cartera idéntica: "
        f"{roles['portfolio_identical_pct']} % de los sorteos.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0, help="solo las primeras N consultas")
    parser.add_argument("--render-only", action="store_true", help="solo regenera el .md")
    args = parser.parse_args()
    if args.render_only:
        DOC.write_text(render_markdown(json.loads(OUT.read_text(encoding="utf-8"))), encoding="utf-8")
        return
    t0 = time.time()
    engine = Recommender(ROOT / "artifacts", enable_semantic=False)
    bank = json.loads((ROOT / "eval/query_bank.json").read_text(encoding="utf-8"))["queries"]
    # Banco completo + consultas de varios gustos en competencia (auditoría y el ejemplo del video).
    queries = [q["query"] for q in bank] + AUDIT_QUERIES + [
        "Busco arquitectura colonial, museos, mezcal y naturaleza, sin playa"]
    queries = list(dict.fromkeys(queries))[: args.limit or None]
    # Una consulta que pide y excluye lo mismo no tiene átomos de gusto: se informa, no se simula.
    contradictory = []
    for query in queries:
        try:
            _extract_facets(engine, query)
        except ValueError:
            contradictory.append(query)
    queries = [q for q in queries if q not in contradictory]

    report = {"protocol": __import__("restmex_recommender.orbit_eval").orbit_eval.__doc__,
              "mode": "léxico (sin E5, igual que audit_multiverse)",
              "generated_at": time.strftime("%Y-%m-%d %H:%M"), "users_per_query": args.users,
              "queries_total": len(queries), "skipped_contradictory": contradictory, "evidence_type": "usuarios simulados (no humanos)",
              "strange_loop": {}, "per_query": {}}
    for population in POPULATIONS:
        rows = []
        for i, query in enumerate(queries):
            row = simulate_users(engine, query, users=args.users, population=population, seed=i)
            if row:
                rows.append(row)
        report["strange_loop"][population] = summarize_users(rows)
        report["strange_loop"][population]["skipped_single_facet"] = len(queries) - len(rows)
        report["per_query"][population] = [
            {"query": r["query"], "facets": r["facets"], "anchor_hit_at_1": r["anchor_hit_at_1"],
             "informativa": r["curves"]["informativa"]["hit_at_1"], "azar": r["curves"]["azar"]["hit_at_1"]}
            for r in rows]
    report["role_sensitivity"] = role_sensitivity(engine, queries)
    report["duration_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
