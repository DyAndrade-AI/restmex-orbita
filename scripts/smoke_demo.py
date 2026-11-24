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


# --- Órbita y Strange Loop: las consultas de la exposición, con comprobaciones automáticas.
ORBITA = {
    "Quiero cascadas, vegetación y aventura al aire libre": ["cascadas", "vegetación", "aventura"],
    "Busco un viaje con niños pequeños y actividades familiares": ["familia"],
    "Quiero arquitectura colonial, gastronomía tradicional y actividades culturales, pero sin playa":
        ["arquitectura colonial", "gastronomía", "culturales"],
    "Me interesan viñedos, vino y recorridos por bodegas": ["vino"],
    "Quiero bosque, cabañas, gastronomía y aventura, sin playa": ["bosque y montaña", "gastronomía", "aventura"],
}
for query, expected in ORBITA.items():
    ranking = engine.recommend(query)
    orbit = engine.recommend_futures(query)
    loop = engine.explore_multiverse(query, futures=20_000)
    atoms = [f["label"] for f in orbit["facets"]]
    towns = [d["town"] for d in orbit["destinations"]]
    print(f"\n{query}\n  Átomos: {atoms}  · excluye: {orbit['excluded_concepts'] or '—'}")
    print("  Top 5:", ", ".join(i["name"] for i in ranking["top"]))
    print("  Bottom 5:", ", ".join(i["name"] for i in ranking["bottom"]))
    for d in orbit["destinations"]:
        cf = d["counterfactual"]
        print(f"  {d['role']:<13} {d['name']:<28} shift={cf['priority_shift']} faceta={cf['facet']} "
              f"[{d['eligibility']}]")
    q = loop["question"]
    print(f"  Strange Loop: {loop['surviving_futures']} futuros · H={loop['decision_entropy']} · "
          f"pregunta: {q['prompt'] if q else 'ninguna (una sola faceta: nada que desempatar)'}")
    assert atoms == expected, atoms
    assert len(ranking["top"]) == len(ranking["bottom"]) == 5
    assert towns[0] == ranking["top"][0]["town"] and len(set(towns)) == len(towns) == 5
    if q:
        assert sum(b["survivors"] for b in q["branches"]) == loop["surviving_futures"]
        branch = q["branches"][0]
        after = engine.explore_multiverse(query, futures=20_000,
                                          choices=[{"keep": branch["keep"], "sacrifice": branch["sacrifice"]}])
        assert after["surviving_futures"] == branch["survivors"]
        print(f"  Colapso '{branch['keep']}' > '{branch['sacrifice']}': {after['surviving_futures']} "
              f"futuros, gana {after['destinations'][0]['name']}")
print("\nDemo OK: todas las comprobaciones pasaron.")
