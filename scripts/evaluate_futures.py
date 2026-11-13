#!/usr/bin/env python3
"""Audita diversidad, estabilidad y contrato de las constelaciones contrafactuales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import QUERIES  # noqa: E402

from restmex_recommender import Recommender  # noqa: E402


def main() -> None:
    engine = Recommender(ROOT / "artifacts", ROOT / "models/multilingual-e5-base")
    profile_rows = [engine.profile_index[(town, "ALL")] for town in engine.towns]
    profiles = engine.lexical_profiles[profile_rows]
    similarities = (profiles @ profiles.T).toarray()
    records = []

    for query in QUERIES:
        result = engine.recommend_futures(query, risk=0.55)
        repeated = engine.recommend_futures(query, risk=0.55)
        towns = [item["town"] for item in result["destinations"]]
        indices = [engine.town_to_idx[town] for town in towns]
        pair_distances = [
            1.0 - float(similarities[left, right])
            for pos, left in enumerate(indices)
            for right in indices[pos + 1 :]
        ]
        records.append(
            {
                "query": query,
                "destinations": towns,
                "unique": len(set(towns)) == 5,
                "deterministic": towns
                == [item["town"] for item in repeated["destinations"]],
                "mean_pairwise_lexical_distance": round(float(np.mean(pair_distances)), 4),
                "mean_resilience": round(
                    float(np.mean([item["resilience"] for item in result["destinations"]])), 2
                ),
            }
        )

    report = {
        "scope": "Auditoría interna de constelaciones; no estima relevancia humana.",
        "queries": len(records),
        "all_contracts_pass": all(item["unique"] and item["deterministic"] for item in records),
        "mean_pairwise_lexical_distance": round(
            float(np.mean([item["mean_pairwise_lexical_distance"] for item in records])), 4
        ),
        "mean_resilience": round(float(np.mean([item["mean_resilience"] for item in records])), 2),
        "results": records,
    }
    output = ROOT / "artifacts/futures_evaluation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
