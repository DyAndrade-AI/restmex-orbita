"""Evidencia por faceta y por restricción, definida sobre tasas de mención del train.

Para un término t y un pueblo d, r(d, t) es la fracción de reseñas de d que contienen t
(``town_term_rates.npz``). Para un grupo de términos G (una faceta o una restricción) se usa
r(d, G) = max_{t∈G} r(d, t), y la referencia m(G) = mediana_d r(d, G) sobre los 40 pueblos.

Clasificación de una faceta positiva:
- ``respaldada``  si r ≥ max(LIFT·m, MIN_SUPPORT)   (el pueblo habla del tema más que lo típico)
- ``débil``       si 0 < r por debajo del umbral
- ``sin evidencia`` si r = 0 (ausencia de evidencia, que NO equivale a incompatibilidad)

Una restricción negativa ("sin playa") se considera con evidencia explícita de incumplimiento
si r ≥ max(LIFT·m, MIN_VIOLATION). Solo esa evidencia explícita activa la restricción dura.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from .text import CONCEPT_EXPANSIONS, QUERY_FILLER, SPANISH_STOP_WORDS, canonical_text

if TYPE_CHECKING:
    from .engine import Recommender

LIFT = 2.0
MIN_SUPPORT = 0.005  # 0.5 % de las reseñas
MIN_VIOLATION = 0.01  # 1 % de las reseñas


def _vocab_terms(engine: "Recommender", words: Sequence[str]) -> List[str]:
    vocab = engine.vectorizer.vocabulary_
    out: List[str] = []
    for word in words:
        for term in engine.analyzer(word):
            if term in vocab and term not in out:
                out.append(term)
    return out


def _concept_group(token: str) -> List[str]:
    """Si el token dispara un concepto turístico, devuelve todos sus disparadores."""
    for triggers in CONCEPT_EXPANSIONS:
        canonical = canonical_text(" ".join(triggers)).split()
        if token in canonical:
            return canonical
    return [token]


def term_groups(engine: "Recommender", text: str, *, expand_concepts: bool) -> List[dict]:
    """Agrupa las palabras de contenido de ``text`` en facetas con sus términos del vocabulario."""
    groups: List[dict] = []
    seen = set()
    for token in canonical_text(text).split():
        if token in SPANISH_STOP_WORDS or token in QUERY_FILLER or len(token) < 3:
            continue
        words = _concept_group(token) if expand_concepts else [token]
        key = tuple(sorted(words))
        if key in seen:
            continue
        terms = _vocab_terms(engine, words)
        if not terms:
            groups.append({"label": token, "terms": [], "known": False})
            seen.add(key)
            continue
        seen.add(key)
        groups.append({"label": token, "terms": terms, "known": True})
    return groups


def group_rates(engine: "Recommender", groups: Sequence[dict]) -> np.ndarray:
    """Matriz pueblos × grupos con r(d, G); ceros si falta el artefacto o el grupo es OOV."""
    out = np.zeros((len(engine.towns), len(groups)), dtype=np.float64)
    for j, group in enumerate(groups):
        table = engine.term_rate_table(group["terms"])
        if table is not None:
            out[:, j] = table.max(axis=1)
    return out


def classify_support(rate: float, median: float) -> str:
    if rate <= 0:
        return "sin evidencia"
    if rate >= max(LIFT * median, MIN_SUPPORT):
        return "respaldada"
    return "débil"


def violation_mask(engine: "Recommender", negative: str) -> Dict[str, object]:
    """Pueblos con evidencia explícita de incumplir alguna restricción negativa."""
    groups = term_groups(engine, negative, expand_concepts=True) if negative.strip() else []
    groups = [g for g in groups if g["known"]]
    mask = np.zeros(len(engine.towns), dtype=bool)
    if not groups:
        return {"groups": [], "rates": np.zeros((len(engine.towns), 0)), "mask": mask}
    rates = group_rates(engine, groups)
    medians = np.median(rates, axis=0)
    thresholds = np.maximum(LIFT * medians, MIN_VIOLATION)
    mask = (rates >= thresholds[None, :]).any(axis=1)
    return {"groups": groups, "rates": rates, "medians": medians, "mask": mask}


def explain_destination(
    idx: int,
    positive_groups: Sequence[dict],
    positive_rates: np.ndarray,
    violations: Dict[str, object],
) -> dict:
    medians = np.median(positive_rates, axis=0) if positive_rates.size else np.zeros(0)
    satisfied, weak, unsupported = [], [], []
    for j, group in enumerate(positive_groups):
        if not group["known"]:
            unsupported.append({"facet": group["label"], "reason": "fuera del vocabulario del train"})
            continue
        rate, median = float(positive_rates[idx, j]), float(medians[j])
        row = {
            "facet": group["label"],
            "rate": round(rate * 100, 2),
            "median_rate": round(median * 100, 2),
        }
        status = classify_support(rate, median)
        (satisfied if status == "respaldada" else weak if status == "débil" else unsupported).append(
            {**row, "reason": status}
        )
    broken = []
    for j, group in enumerate(violations["groups"]):
        rate = float(violations["rates"][idx, j])
        median = float(violations["medians"][j])
        if rate >= max(LIFT * median, MIN_VIOLATION):
            broken.append(
                {
                    "constraint": f"sin {group['label']}",
                    "rate": round(rate * 100, 2),
                    "median_rate": round(median * 100, 2),
                }
            )
    return {"satisfied": satisfied, "weak": weak, "unsupported": unsupported, "violated": broken}


def narrative(item: dict, least: bool) -> str:
    """Explicación en una frase, construida solo con los campos calculados."""
    parts = []
    if item["violated"]:
        names = ", ".join(v["constraint"] for v in item["violated"])
        parts.append(f"Incumple {names}: sus reseñas lo mencionan explícitamente.")
    if item["satisfied"]:
        parts.append("Respaldo claro en: " + ", ".join(s["facet"] for s in item["satisfied"]) + ".")
    if item["weak"]:
        parts.append("Respaldo débil en: " + ", ".join(s["facet"] for s in item["weak"]) + ".")
    if item["unsupported"]:
        parts.append(
            "Sin evidencia en el train para: "
            + ", ".join(s["facet"] for s in item["unsupported"])
            + " (ausencia de menciones, no prueba de incompatibilidad)."
        )
    if not parts:
        parts.append("La afinidad proviene de similitud general de vocabulario.")
    if least and not item["violated"]:
        parts.append("Queda abajo por menor afinidad relativa, no por baja calidad.")
    return " ".join(parts)
