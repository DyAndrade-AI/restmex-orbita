#!/usr/bin/env python3
"""Copia las cifras que muestra la web desde artifacts/ a web/src/data/metrics.json.

El build de la web (Docker o Netlify) solo ve la carpeta web/, así que la página no puede leer
artifacts/ directamente. Correr después de evaluate, validate-orbit o audit-multiverse."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/src/data/metrics.json"


def load(name: str) -> dict:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def main() -> None:
    report = load("evaluation_report.json")
    gold, holdout = report["gold"]["aggregate"], report["holdout"]
    orbit = load("orbit_validation.json")
    futures = load("futures_evaluation_report.json")
    audit = load("multiverse_audit.json")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "--co"], cwd=ROOT,
                           capture_output=True, text=True).stdout
    test_count = sum(1 for line in tests.splitlines() if "::" in line)
    prior, concentrated = orbit["strange_loop"]["prior"], orbit["strange_loop"]["concentrada"]
    roles = orbit["role_sensitivity"]
    metrics = {
        "generated_from": "artifacts/*.json vía scripts/export_web_metrics.py",
        "corpus": {"reviews": 208051, "towns": 40},
        "gold": {
            "queries": gold["queries"], "wide_queries": gold["wide_queries"],
            "precision_at_5": gold["precision_at_5_wide"], "hit_at_1": gold["hit_at_1"],
            "ndcg_at_5": gold["ndcg_at_5"], "mode": report["mode"],
        },
        "holdout": {
            "reviews": holdout["evaluated_queries"], "hit_at_5": holdout["metrics"]["hit_at_5"],
            "hit_at_1": holdout["metrics"]["hit_at_1"], "mrr": holdout["metrics"]["mrr"],
            "random_hit_at_5": holdout["random_expectation"]["hit_at_5"],
            "popularity_hit_at_5": holdout["popularity_baseline"]["hit_at_5"],
            "bottom_error_rate": holdout["metrics"]["bottom_error_rate"],
        },
        "simulated_users": {
            "queries": prior["queries"],
            "anchor_hit_at_1": prior["anchor_hit_at_1"],
            "loop_hit_at_1": prior["informativa"]["hit_at_1"],
            "random_hit_at_1": prior["azar"]["hit_at_1"],
            "gain_3q": prior["hit_at_1_gain"]["3"],
            "gain_1q": prior["hit_at_1_gain"]["1"],
            "concentrated": {
                "anchor_hit_at_1": concentrated["anchor_hit_at_1"],
                "loop_hit_at_1": concentrated["informativa"]["hit_at_1"],
                "random_hit_at_1": concentrated["azar"]["hit_at_1"],
                "gain_3q": concentrated["hit_at_1_gain"]["3"],
            },
        },
        # Poblaciones que rompen un supuesto del modelo (resultados adversos incluidos).
        "stress": {
            name: {
                "queries": row["queries"],
                "loop_hit_at_1": row["informativa"]["hit_at_1"],
                "gain_3q": row["hit_at_1_gain"]["3"],
                "info_minus_random_1q": row["hit_at_1_gain"]["1"],
            }
            for name, row in orbit["strange_loop"].items()
        },
        "role_sensitivity": {
            "spread": roles["spread"], "role_unchanged_pct": roles["role_unchanged_pct"],
            "portfolio_jaccard_mean": roles["portfolio_jaccard_mean"],
            "portfolio_identical_pct": roles["portfolio_identical_pct"],
        },
        "orbit_audit": {
            "queries": futures["queries"], "all_contracts_pass": futures["all_contracts_pass"],
            "mean_resilience": futures["mean_resilience"],
        },
        "monte_carlo": {
            "max_diff_pp_at_50k": max(x["max_abs_diff_pp"] for x in audit["convergence"]
                                      if x["futures"] == 50_000),
            "seed_agreement_min": min(x["top_destination_agreement"] for x in audit["seeds"]),
        },
        "tests": test_count,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {test_count} pruebas")


if __name__ == "__main__":
    main()
