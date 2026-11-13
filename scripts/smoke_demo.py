#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender import Recommender  # noqa: E402

engine = Recommender(ROOT / "artifacts", ROOT / "models/multilingual-e5-base")
queries = [
    "Quiero playas tranquilas, agua clara y actividades de snorkel",
    "Busco arquitectura colonial, museos y comida tradicional",
    "Quiero naturaleza, montaña y senderismo, sin playa ni vida nocturna",
]
for query in queries:
    result = engine.recommend(query)
    print(f"\n{query}\n  Modo: {result['mode']}")
    print("  Top 5:", ", ".join(item["name"] for item in result["top"]))
    print("  Bottom 5:", ", ".join(item["name"] for item in result["bottom"]))

