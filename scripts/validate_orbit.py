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
from restmex_recommender.orbit_eval import (  # noqa: E402
    POPULATIONS, role_sensitivity, simulate_users, summarize_users,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0, help="solo las primeras N consultas")
    args = parser.parse_args()
    t0 = time.time()
    engine = Recommender(ROOT / "artifacts", enable_semantic=False)
    bank = json.loads((ROOT / "eval/query_bank.json").read_text(encoding="utf-8"))["queries"]
    # Banco completo + consultas de varios gustos en competencia (auditoría y el ejemplo del video).
    queries = [q["query"] for q in bank] + AUDIT_QUERIES + [
        "Busco arquitectura colonial, museos, mezcal y naturaleza, sin playa"]
    queries = list(dict.fromkeys(queries))[: args.limit or None]

    report = {"protocol": __import__("restmex_recommender.orbit_eval").orbit_eval.__doc__,
              "mode": "léxico (sin E5, igual que audit_multiverse)",
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
    out = ROOT / "artifacts/orbit_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
