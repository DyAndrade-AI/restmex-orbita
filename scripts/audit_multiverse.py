#!/usr/bin/env python3
"""Auditoría estadística de Strange Loop: convergencia en N, semillas, riesgo y coherencia."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender import Recommender  # noqa: E402

QUERIES = [
    "Quiero arquitectura colonial, gastronomía tradicional y actividades culturales, pero sin playa",
    "Quiero bosque, cabañas, gastronomía y aventura",
    "Playas tranquilas, agua turquesa, snorkel y comida de mar",
    "Cenotes, comida yucateca, historia y tranquilidad",
    "Desierto, silencio, paisajes naturales y vino",
]


def shares(result: dict, towns) -> np.ndarray:
    # Vector completo: comparar solo las 5 cuencas visibles crea saltos falsos cuando el 5.º y el
    # 6.º lugar están casi empatados.
    return np.array([result["all_shares"].get(t, 0.0) for t in towns])


def main() -> None:
    engine = Recommender(ROOT / "artifacts", enable_semantic=False)
    towns = engine.towns
    report = {"convergence": [], "seeds": [], "risk": [], "coherence": []}
    for query in QUERIES:
        reference = shares(engine.explore_multiverse(query, futures=200_000, seed=1), towns)
        for n in (1_000, 5_000, 10_000, 50_000, 100_000):
            diff = np.abs(shares(engine.explore_multiverse(query, futures=n, seed=2), towns) - reference)
            report["convergence"].append({"query": query[:40], "futures": n,
                                          "max_abs_diff_pp": round(float(diff.max()), 3)})
        runs = [engine.explore_multiverse(query, futures=50_000, seed=s) for s in range(10)]
        top = [r["destinations"][0]["town"] for r in runs]
        matrix = np.vstack([shares(r, towns) for r in runs])
        report["seeds"].append({
            "query": query[:40],
            "top_destination_agreement": round(top.count(max(set(top), key=top.count)) / len(top), 2),
            "max_share_sd_pp": round(float(matrix.std(axis=0).max()), 3),
            "entropy_sd": round(float(np.std([r["decision_entropy"] for r in runs])), 3),
        })
        for risk in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = engine.explore_multiverse(query, risk=risk, futures=50_000)
            report["risk"].append({"query": query[:40], "risk": risk,
                                   "entropy": r["decision_entropy"],
                                   "top": r["destinations"][0]["name"],
                                   "top_share": r["destinations"][0]["share"]})
        classic = engine.recommend(query)
        r = engine.explore_multiverse(query, futures=50_000)
        classic_top = {x["town"] for x in classic["top"]}
        report["coherence"].append({
            "query": query[:40],
            "multiverse_top5_in_classic_top5": len(classic_top & {d["town"] for d in r["destinations"]}),
            "classic_bottom5_win_share_pp": round(sum(
                b["wins_pct"] for b in r["bottom_in_multiverse"]), 3),
        })
    out = ROOT / "artifacts/multiverse_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
