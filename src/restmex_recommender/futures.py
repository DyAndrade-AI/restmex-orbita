"""Constelaciones contrafactuales para explorar más de un futuro plausible."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .engine import _rank_order, _scale
from .text import (
    CONCEPT_EXPANSIONS,
    QUERY_FILLER,
    SPANISH_STOP_WORDS,
    canonical_text,
    expand_for_lexical,
)

if TYPE_CHECKING:
    from .engine import Recommender


ROLE_COPY = (
    ("Ancla", "Tu coincidencia más directa"),
    ("Superviviente", "Resiste aunque una prioridad desaparezca"),
    ("Portal", "Se aleja de lo obvio sin romper tu intención"),
    ("Especialista", "Lleva una de tus prioridades al extremo"),
    ("Comodín", "Equilibra afinidad, estabilidad y sorpresa"),
)



ROLE_DEFINITIONS = {
    "Ancla": "argmax_d r(d): máxima compatibilidad del ranking 5 + 5 (coincide con el top 1).",
    "Superviviente": (
        "argmax 0.38·r + 0.42·R + 0.20·W, con R(d) = media y W(d) = mínimo de la utilidad de rango "
        "al eliminar cada faceta (leave-one-facet-out)."
    ),
    "Portal": (
        "argmax (1−0.62ρ)·r + 0.62ρ·(0.58·dist_ancla + 0.42·max_f M[f,d]) entre destinos con "
        "r ≥ cuantil(0.60 − 0.35ρ) y sin evidencia de incumplir restricciones (ρ = riesgo)."
    ),
    "Especialista": "argmax 0.30·r + 0.70·escala(max_f M[f,d] − media_f M[f,d]).",
    "Comodín": "argmax 0.24·r + 0.30·R + 0.24·escala(min_f M[f,d]) + 0.22·dist_ancla.",
    "diversidad": (
        "Cada rol tras el Ancla se penaliza con 0.18ρ·(distancia media a los ya elegidos), "
        "distancia = 1 − coseno entre perfiles TF-IDF."
    ),
    "estabilidad": (
        "Fracción de 2,000 perturbaciones w ~ Dirichlet(20/K·1) de los pesos de faceta en las que "
        "el destino queda en el top 10 de 0.5·r + 0.5·w·M."
    ),
}


# Intenciones que no son un sustantivo concreto: sus sinónimos deben formar UN átomo (antes
# "niños" y "familia" eran dos gustos que "competían" sin competir). Solo afecta a Órbita y
# Strange Loop; el ranking 5 + 5 sigue usando CONCEPT_EXPANSIONS sin cambios.
FACET_CONCEPTS = (
    ("romántico", ("romantico", "romantica", "romanticos", "pareja", "aniversario", "enamorados",
                   "novia", "novio"), "romantico pareja aniversario cena atardecer"),
    ("familia", ("familia", "familiar", "familias", "ninos", "nino", "ninas", "hijos", "infantil"),
     "familia familiar ninos juegos parque infantil"),
    ("vida nocturna", ("nocturna", "nocturno", "bar", "bares", "antro", "antros", "fiesta",
                       "fiestas", "noche"), "vida nocturna bar bares antro fiesta musica noche"),
    ("descanso", ("tranquilo", "tranquila", "tranquilos", "tranquilas", "tranquilidad", "descansar",
                  "descanso", "relajarse", "relajante", "relajacion", "desconectar",
                  "desconectarme", "paz"), "tranquilo tranquilidad descanso relajarse paz silencio"),
    ("vistas", ("fotografia", "fotos", "foto", "mirador", "miradores", "vistas", "panoramica"),
     "fotografia fotos mirador vista vistas panoramica"),
    ("aventura", ("aventura", "aventuras", "extrema", "extremo", "extremos", "tirolesa", "rappel",
                  "cuatrimoto", "cuatrimotos", "escalada", "canopy"),
     "aventura tirolesa rappel cuatrimoto escalada extremo"),
    ("lago y río", ("lago", "lagos", "laguna", "lagunas", "lancha", "lanchas", "rio", "rios"),
     "lago laguna lancha rio paseo agua"),
    ("café", ("cafe", "cafetal", "cafetales"), "cafe cafetal cafetales finca"),
    ("artesanía", ("artesania", "artesanias", "artesanal", "artesanales", "alfareria", "textiles"),
     "artesania artesanias artesanal alfareria textiles"),
)
# Modificadores que el IDF trataba como gustos propios ("pequeños", "vida", "altura").
FACET_FILLER = frozenset(
    "algo lugar lugares vida pequenos pequenas actividades actividad altura nuestro nuestra "
    "mucho mucha muy buen buena".split()
)

# Pesos de diseño de cada rol (un solo lugar para auditarlos; ver orbit_eval.role_sensitivity).
ROLE_WEIGHTS = {
    "Superviviente": {"relevance": 0.38, "robust": 0.42, "worst": 0.20},
    "Portal": {"mix": 0.62, "distance": 0.58, "peak": 0.42},
    "Especialista": {"relevance": 0.30, "specialization": 0.70},
    "Comodín": {"relevance": 0.24, "robust": 0.30, "balance": 0.24, "distance": 0.22},
    "diversidad": {"penalty": 0.18},
}


def _role_stability(matrix: np.ndarray, relevance: np.ndarray, selected: Sequence[int],
                    draws: int = 2_000, seed: int = 11) -> List[float]:
    """Comprobación de robustez con perturbaciones controladas de los pesos de faceta."""
    rng = np.random.default_rng(seed)
    k = matrix.shape[0]
    weights = rng.dirichlet(np.full(k, 20.0 / k), size=draws)
    scores = 0.5 * relevance[None, :] + 0.5 * (weights @ matrix)
    top10 = np.argsort(-scores, axis=1)[:, :10]
    return [round(float(np.mean((top10 == d).any(axis=1))) * 100.0, 1) for d in selected]


def _extract_facets(engine: "Recommender", text: str, limit: int = 5) -> List[dict]:
    """Convierte una consulta en conceptos de dominio y términos raros del corpus."""
    canonical = canonical_text(text)
    tokens = set(canonical.split())
    consumed = set()
    facets: List[dict] = []

    for triggers, expansion in CONCEPT_EXPANSIONS.items():
        trigger_tokens = set(canonical_text(" ".join(triggers)).split())
        matches = sorted(tokens.intersection(trigger_tokens))
        if not matches:
            continue
        consumed.update(matches)
        facets.append(
            {
                "label": " / ".join(matches[:3]),
                "probe": expansion,
                "source": "concepto turístico",
            }
        )

    for label, triggers, expansion in FACET_CONCEPTS:
        matches = tokens.intersection(triggers) - consumed
        if matches:
            consumed.update(matches)
            facets.append({"label": label, "probe": expansion, "source": "concepto turístico"})

    candidates = []
    vocabulary = engine.vectorizer.vocabulary_
    idf = engine.vectorizer.idf_
    for token in re.findall(r"[a-záéíóúüñ]{3,}", canonical):
        if (
            token in consumed
            or token in SPANISH_STOP_WORDS
            or token in QUERY_FILLER
            or token in FACET_FILLER
            or token not in vocabulary
        ):
            continue
        candidates.append((float(idf[vocabulary[token]]), token))

    seen = {item["label"] for item in facets}
    for _, token in sorted(set(candidates), key=lambda item: (-item[0], item[1])):
        if token not in seen:
            facets.append({"label": token, "probe": token, "source": "señal del usuario"})
            seen.add(token)
        if len(facets) >= limit:
            break

    if not facets:
        facets.append({"label": canonical, "probe": canonical, "source": "intención completa"})
    return facets[:limit]


def _facet_scores(
    engine: "Recommender", facets: Sequence[dict], preferred_types: Sequence[str]
) -> np.ndarray:
    rows = []
    for facet in facets:
        query_vector = engine.vectorizer.transform([expand_for_lexical(facet["probe"])])
        lexical = engine._profile_scores(query_vector, preferred_types)
        evidence, _ = engine._evidence_scores(query_vector)
        rows.append(0.68 * _scale(lexical) + 0.32 * _scale(evidence))
    return np.vstack(rows).astype(np.float32)


def _rank_utility(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return 1.0 - ranks / max(len(values) - 1, 1)


def _scenario_metrics(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula el olvido de cada faceta y devuelve media, peor caso y volatilidad."""
    if matrix.shape[0] == 1:
        utilities = _rank_utility(matrix[0])[None, :]
    else:
        scenarios = [np.delete(matrix, idx, axis=0).mean(axis=0) for idx in range(matrix.shape[0])]
        utilities = np.vstack([_rank_utility(row) for row in scenarios])
    robust = utilities.mean(axis=0)
    worst = utilities.min(axis=0)
    volatility = utilities.std(axis=0)
    return robust, worst, volatility


def _pick(
    role_score: np.ndarray,
    selected: List[int],
    distances: np.ndarray,
    risk: float,
    eligible: Optional[np.ndarray] = None,
    penalty: float = ROLE_WEIGHTS["diversidad"]["penalty"],
) -> int:
    score = role_score.copy()
    if selected:
        diversity = distances[:, selected].mean(axis=1)
        score = (1.0 - penalty * risk) * score + penalty * risk * diversity
        score[selected] = -np.inf
    if eligible is not None and (eligible & np.isfinite(score)).any():
        score[~eligible] = -np.inf  # si nadie es elegible se ignora el filtro en vez de repetir
    return int(np.argmax(score))


def _counterfactual(
    matrix: np.ndarray, facets: Sequence[dict], anchor: int, destination: int
) -> Dict[str, object]:
    differences = matrix[:, destination] - matrix[:, anchor]
    facet_idx = int(np.argmax(differences))
    mean_difference = float(differences.mean())
    best_difference = float(differences[facet_idx])
    if best_difference > 1e-7 and mean_difference < 0:
        shift = -mean_difference / (best_difference - mean_difference)
        shift_percent = int(np.clip(np.ceil(shift * 100.0), 1, 100))
    elif best_difference > 1e-7:
        shift_percent = 0
    else:
        shift_percent = None
    return {
        "facet": facets[facet_idx]["label"],
        "priority_shift": shift_percent,
        "advantage": round(best_difference * 100.0, 1),
    }


def constellation_inputs(engine: "Recommender", query: str, risk: float) -> dict:
    """Todo lo que la selección de roles necesita, calculado una vez por consulta."""
    scored = engine._score_query(query)
    relevance = scored["scores"]
    facets = _extract_facets(engine, scored["intent"].positive)
    matrix = _facet_scores(engine, facets, scored["intent"].preferred_types)
    robust, worst, volatility = _scenario_metrics(matrix)

    profile_rows = [engine.profile_index[(town, "ALL")] for town in engine.towns]
    profiles = engine.lexical_profiles[profile_rows]
    similarities = (profiles @ profiles.T).toarray().astype(np.float32)
    distances = np.clip(1.0 - similarities, 0.0, 1.0)
    anchor = int(np.argmax(relevance))
    threshold = float(np.quantile(relevance, 0.60 - 0.35 * risk))
    # Frontera negativa = evidencia explícita de incumplimiento (explain.violation_mask), la
    # misma regla que el ranking 5 + 5; ya no un cuantil fijo que excluía siempre 14 pueblos.
    unsafe = scored["violations"]["mask"]
    eligible = ~unsafe & (relevance >= threshold)
    eligible[anchor] = True
    return {
        "scored": scored, "facets": facets, "matrix": matrix, "relevance": relevance,
        "robust": robust, "worst": worst, "volatility": volatility, "distances": distances,
        "anchor": anchor, "unsafe": unsafe, "eligible": eligible,
        "distance_from_anchor": _scale(distances[:, anchor]),
        "specialization": _scale(matrix.max(axis=0) - matrix.mean(axis=0)),
        "balance": _scale(matrix.min(axis=0)),
        "peak": matrix.max(axis=0),
    }


def select_roles(x: dict, risk: float, weights: Optional[dict] = None, count: int = 5) -> List[int]:
    """Índices de destino por rol (Ancla primero) para unos pesos de rol dados."""
    w = weights or ROLE_WEIGHTS
    r, dist = x["relevance"], x["distance_from_anchor"]
    sv, po, es, co = w["Superviviente"], w["Portal"], w["Especialista"], w["Comodín"]
    role_scores = [
        r,
        sv["relevance"] * r + sv["robust"] * x["robust"] + sv["worst"] * x["worst"],
        (1.0 - po["mix"] * risk) * r
        + po["mix"] * risk * (po["distance"] * dist + po["peak"] * x["peak"]),
        es["relevance"] * r + es["specialization"] * x["specialization"],
        co["relevance"] * r + co["robust"] * x["robust"] + co["balance"] * x["balance"]
        + co["distance"] * dist,
    ]
    selected = [x["anchor"]]
    for role_idx in range(1, count):
        selected.append(_pick(role_scores[role_idx], selected, x["distances"], risk,
                              x["eligible"], w["diversidad"]["penalty"]))
    return selected


def build_future_constellation(
    engine: "Recommender", query: str, risk: float = 0.55, count: int = 5
) -> dict:
    if not 0.0 <= risk <= 1.0:
        raise ValueError("El riesgo debe estar entre 0 y 1.")
    if not 1 <= count <= len(ROLE_COPY):
        raise ValueError(f"La constelación admite entre 1 y {len(ROLE_COPY)} destinos.")

    x = constellation_inputs(engine, query, risk)
    scored, facets, matrix, relevance = x["scored"], x["facets"], x["matrix"], x["relevance"]
    intent, anchor, unsafe = scored["intent"], x["anchor"], x["unsafe"]
    robust, volatility, distance_from_anchor = x["robust"], x["volatility"], x["distance_from_anchor"]
    selected = select_roles(x, risk, count=count)

    results = []
    for role_idx, destination_idx in enumerate(selected):
        item = engine.destinations[destination_idx]
        evidence_idx = int(scored["best_evidence"][destination_idx])
        counterfactual = _counterfactual(matrix, facets, anchor, destination_idx)
        if role_idx == 0:
            counterfactual = {
                "facet": facets[int(np.argmax(matrix[:, destination_idx]))]["label"],
                "priority_shift": 0,
                "advantage": 0.0,
            }
        results.append(
            {
                "role": ROLE_COPY[role_idx][0],
                "role_explanation": ROLE_COPY[role_idx][1],
                "town": item["town"],
                "name": item["display_name"],
                "region": item["display_region"],
                "affinity": round(float(relevance[destination_idx] * 100.0), 1),
                "resilience": round(float(robust[destination_idx] * 100.0), 1),
                "fragility": round(float(volatility[destination_idx] * 100.0), 1),
                "distance_from_anchor": round(float(distance_from_anchor[destination_idx] * 100.0), 1),
                "dominant_facet": facets[int(np.argmax(matrix[:, destination_idx]))]["label"],
                "counterfactual": counterfactual,
                "matched_terms": engine._matched_terms(scored["query_vector"], item["town"]),
                "keywords": item["keywords"]["ALL"][:6],
                "evidence": engine.evidence[evidence_idx] if evidence_idx >= 0 else None,
                "review_count": item["review_count"],
            }
        )

    order = _rank_order(relevance)
    bottom_view = []
    for idx in order[::-1][:5]:
        idx = int(idx)
        item = engine.destinations[idx]
        if unsafe[idx]:
            verdict = "Imposible bajo las restricciones: incumple una exclusión con evidencia explícita."
            cf = {"facet": None, "priority_shift": None, "advantage": 0.0}
        else:
            cf = _counterfactual(matrix, facets, anchor, idx)
            if cf["priority_shift"] is None:
                verdict = ("Ninguna redistribución de atención entre las facetas lo hace superar al "
                           "Ancla: es más débil en todas.")
            elif cf["priority_shift"] == 0:
                verdict = f"Ya supera al Ancla si solo importa {cf['facet']}."
            else:
                verdict = (f"Tendría que moverse {cf['priority_shift']} % de la atención hacia "
                           f"{cf['facet']} para superar al Ancla.")
        bottom_view.append({
            "town": item["town"], "name": item["display_name"],
            "global_rank": int(np.flatnonzero(order == idx)[0]) + 1,
            "affinity": round(float(relevance[idx] * 100.0), 1),
            "counterfactual": cf, "verdict": verdict,
        })
    stability = _role_stability(matrix, relevance, selected)
    for row, value in zip(results, stability):
        row["stability"] = value

    return {
        "query": intent.original,
        "mode": "constelación contrafactual",
        "bottom_perspective": bottom_view,
        "base_mode": scored["mode"],
        "risk": round(float(risk), 2),
        "facets": [{"label": item["label"], "source": item["source"]} for item in facets],
        "excluded_concepts": intent.negative,
        "destinations": results,
        "method": {
            "definitions": ROLE_DEFINITIONS,
            "scenarios": len(facets) if len(facets) > 1 else 1,
            "inventory": len(engine.towns),
            "deterministic": True,
            "negative_boundary": bool(intent.negative),
        },
    }


def _entropy(winners: np.ndarray, town_count: int) -> float:
    if not len(winners):
        return 0.0
    counts = np.bincount(winners, minlength=town_count).astype(np.float64)
    probabilities = counts[counts > 0] / len(winners)
    value = -float(np.sum(probabilities * np.log(probabilities)))
    return value / max(float(np.log(town_count)), 1e-9)


def _top_winner(winners: np.ndarray, town_count: int) -> Tuple[int, int]:
    counts = np.bincount(winners, minlength=town_count)
    index = int(np.argmax(counts))
    return index, int(counts[index])


def _sacrifice_question(
    engine: "Recommender",
    winners: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
    facets: Sequence[dict],
    choices: Sequence[dict],
) -> Optional[dict]:
    active_indices = np.flatnonzero(active)
    active_winners = winners[active_indices]
    if len(active_indices) < 300 or len(facets) < 2:
        return None

    current_entropy = _entropy(active_winners, len(engine.towns))
    used_pairs = {
        frozenset((str(choice.get("keep")), str(choice.get("sacrifice"))))
        for choice in choices
    }
    best = None
    for left in range(len(facets)):
        for right in range(left + 1, len(facets)):
            labels = frozenset((facets[left]["label"], facets[right]["label"]))
            if labels in used_pairs:
                continue
            prefer_left = weights[active_indices, left] >= weights[active_indices, right]
            left_count = int(prefer_left.sum())
            right_count = int(len(active_indices) - left_count)
            if min(left_count, right_count) < 0.12 * len(active_indices):
                continue
            left_entropy = _entropy(active_winners[prefer_left], len(engine.towns))
            right_entropy = _entropy(active_winners[~prefer_left], len(engine.towns))
            conditional = (
                left_count * left_entropy + right_count * right_entropy
            ) / len(active_indices)
            gain = max(current_entropy - conditional, 0.0)
            balance = 1.0 - abs(left_count / len(active_indices) - 0.5)
            score = gain * (0.75 + 0.25 * balance)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "gain": gain,
                    "left": left,
                    "right": right,
                    "left_mask": prefer_left,
                    "left_count": left_count,
                    "right_count": right_count,
                }

    if best is None:
        return None

    def branch(keep_idx: int, sacrifice_idx: int, local_mask: np.ndarray) -> dict:
        branch_winners = active_winners[local_mask]
        destination, count = _top_winner(branch_winners, len(engine.towns))
        return {
            "keep": facets[keep_idx]["label"],
            "sacrifice": facets[sacrifice_idx]["label"],
            "survivors": int(len(branch_winners)),
            "eliminated": int(len(active_indices) - len(branch_winners)),
            "destination": engine.destinations[destination]["display_name"],
            "destination_share": round(100.0 * count / max(len(branch_winners), 1), 1),
        }

    left = int(best["left"])
    right = int(best["right"])
    return {
        "prompt": (
            f"Para cerrar más líneas temporales: ¿conservarías {facets[left]['label']} "
            f"o {facets[right]['label']}?"
        ),
        "information_gain": round(float(best["gain"] * 100.0), 2),
        "branches": [
            branch(left, right, best["left_mask"]),
            branch(right, left, ~best["left_mask"]),
        ],
    }


def _speculative_trajectory(
    engine: "Recommender",
    matrix: np.ndarray,
    relevance: np.ndarray,
    preference: np.ndarray,
    facets: Sequence[dict],
    steps: int = 3,
) -> List[dict]:
    current = preference.astype(np.float32).copy()
    selected: List[int] = []
    trajectory = []
    for step in range(steps):
        scores = 0.78 * (current @ matrix) + 0.22 * relevance
        if selected:
            scores[selected] = -np.inf
        destination = int(np.argmax(scores))
        item = engine.destinations[destination]
        dominant = int(np.argmax(matrix[:, destination]))
        trajectory.append(
            {
                "step": step + 1,
                "name": item["display_name"],
                "region": item["display_region"],
                "awakens": facets[dominant]["label"],
            }
        )
        selected.append(destination)
        destination_taste = matrix[:, destination] + 0.03
        destination_taste /= destination_taste.sum()
        current = 0.72 * current + 0.28 * destination_taste
        current /= current.sum()
    return trajectory


def multiverse_core(
    engine: "Recommender", query: str, risk: float, futures: int, seed: Optional[int] = None
) -> dict:
    """Muestreo de Strange Loop sin decisiones: pesos, utilidades y ganador por futuro."""
    scored = engine._score_query(query)
    intent = scored["intent"]
    relevance = scored["scores"]
    facets = _extract_facets(engine, intent.positive)
    matrix = _facet_scores(engine, facets, intent.preferred_types)

    seed_material = f"{intent.original}|{risk:.4f}|strange-loop-v1".encode("utf-8")
    if seed is None:
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    concentration = 28.0 - 24.0 * risk
    alpha = np.full(len(facets), concentration / len(facets), dtype=np.float64)
    weights = generator.dirichlet(alpha, size=futures).astype(np.float32)

    possible_scores = 0.16 * relevance[None, :] + 0.84 * (weights @ matrix)
    # Cada línea temporal también incluye variación no observada de la experiencia. El ruido
    # Gumbel convierte la afinidad en una elección probabilística sin perder reproducibilidad.
    experience_scale = 0.015 + 0.065 * risk
    possible_scores += generator.gumbel(
        loc=0.0,
        scale=experience_scale,
        size=possible_scores.shape,
    ).astype(np.float32)
    unsafe = scored["violations"]["mask"]
    possible_scores[:, unsafe] = -1e6  # restricción dura con evidencia explícita

    winners = np.argmax(possible_scores, axis=1).astype(np.int16)
    return {
        "scored": scored, "facets": facets, "matrix": matrix, "relevance": relevance,
        "weights": weights, "possible_scores": possible_scores, "winners": winners,
        "unsafe": unsafe, "seed": seed,
    }


def simulate_multiverse(
    engine: "Recommender",
    query: str,
    risk: float = 0.55,
    futures: int = 50_000,
    choices: Optional[Sequence[dict]] = None,
    seed: Optional[int] = None,
) -> dict:
    """Muestrea posibles ponderaciones del gusto y encuentra sus destinos ganadores.

    ``seed`` solo se usa en auditorías de variabilidad; por defecto la semilla se deriva de la
    consulta y el riesgo para que la misma pregunta dé siempre el mismo multiverso."""
    if not 0.0 <= risk <= 1.0:
        raise ValueError("El riesgo debe estar entre 0 y 1.")
    if not 1_000 <= futures <= 200_000:
        raise ValueError("La simulación admite entre 1,000 y 200,000 futuros.")
    choices = list(choices or [])
    if len(choices) > 4:
        raise ValueError("Sólo se permiten cuatro decisiones de colapso por simulación.")

    core = multiverse_core(engine, query, risk, futures, seed)
    scored, facets, matrix, relevance = core["scored"], core["facets"], core["matrix"], core["relevance"]
    intent, weights, winners, unsafe = scored["intent"], core["weights"], core["winners"], core["unsafe"]
    possible_scores, seed = core["possible_scores"], core["seed"]
    facet_labels = [item["label"] for item in facets]
    active = np.ones(futures, dtype=bool)
    normalized_choices = []
    for choice in choices:
        keep = str(choice.get("keep", ""))
        sacrifice = str(choice.get("sacrifice", ""))
        if keep not in facet_labels or sacrifice not in facet_labels or keep == sacrifice:
            raise ValueError("La decisión de sacrificio no coincide con las facetas activas.")
        keep_idx = facet_labels.index(keep)
        sacrifice_idx = facet_labels.index(sacrifice)
        active &= weights[:, keep_idx] >= weights[:, sacrifice_idx]
        normalized_choices.append({"keep": keep, "sacrifice": sacrifice})

    active_indices = np.flatnonzero(active)
    if len(active_indices) < 50:
        raise ValueError("Tus decisiones cerraron casi todos los futuros; reinicia el multiverso.")
    active_winners = winners[active_indices]
    counts = np.bincount(active_winners, minlength=len(engine.towns))
    order = np.argsort(-counts, kind="stable")
    anchor = int(order[0])

    destinations = []
    for destination in order[:5]:
        count = int(counts[destination])
        if count == 0:
            continue
        local = active_indices[active_winners == destination]
        mean_taste = weights[local].mean(axis=0)
        identity_order = np.argsort(-mean_taste)[:2]
        item = engine.destinations[int(destination)]
        evidence_idx = int(scored["best_evidence"][destination])
        counterfactual = _counterfactual(matrix, facets, anchor, int(destination))
        p_hat = count / len(active_indices)
        se = float(np.sqrt(p_hat * (1 - p_hat) / len(active_indices)))
        destinations.append(
            {
                "town": item["town"],
                "name": item["display_name"],
                "region": item["display_region"],
                "count": count,
                "share": round(100.0 * p_hat, 2),
                "share_se": round(100.0 * se, 3),
                "share_ci95": [round(100.0 * max(0.0, p_hat - 1.96 * se), 2),
                               round(100.0 * min(1.0, p_hat + 1.96 * se), 2)],
                "identity": (
                    f"La versión de ti que prioriza {facets[int(identity_order[0])]['label']} "
                    f"y {facets[int(identity_order[1])]['label']}"
                    if len(identity_order) > 1
                    else f"La versión de ti que prioriza {facets[int(identity_order[0])]['label']}"
                ),
                "dominant_facet": facets[int(identity_order[0])]["label"],
                "sacrifice": counterfactual,
                "affinity": round(float(relevance[destination] * 100.0), 1),
                "evidence": engine.evidence[evidence_idx] if evidence_idx >= 0 else None,
            }
        )

    # --- Diagnóstico de Monte Carlo: ¿cambiaría la conclusión con menos simulaciones?
    n_active = len(active_indices)
    convergence = []
    for fraction in (0.1, 0.5, 1.0):
        subset = active_winners[: max(1, int(n_active * fraction))]
        sub_counts = np.bincount(subset, minlength=len(engine.towns)) / len(subset)
        convergence.append({
            "futures": int(len(subset)),
            "top_destination": engine.destinations[int(np.argmax(sub_counts))]["display_name"],
            "max_abs_share_diff_vs_full_pp": round(
                float(np.abs(sub_counts - counts / n_active).max()) * 100.0, 3),
        })
    monte_carlo = {
        "estimator": "proporción de escenarios ganados, p̂ = k/N; EE = sqrt(p̂(1−p̂)/N)",
        "worst_case_se_pp": round(100.0 * 0.5 / np.sqrt(n_active), 3),
        "convergence": convergence,
        "note": (
            "Frecuencia simulada ≠ probabilidad real de satisfacción: es la fracción de "
            "ponderaciones posibles de tus gustos en las que cada destino resulta el mejor."
        ),
    }

    # --- Bottom 5 en el multiverso: ¿siguen siendo malas coincidencias en otros futuros?
    base_order = _rank_order(relevance)
    active_scores = possible_scores[active_indices]
    ranks = np.empty_like(active_scores, dtype=np.int16)
    ranks[np.arange(n_active)[:, None], np.argsort(-active_scores, axis=1)] = np.arange(
        len(engine.towns), dtype=np.int16)
    bottom_analysis = []
    for idx in base_order[::-1][:5]:
        idx = int(idx)
        item = engine.destinations[idx]
        if unsafe[idx]:
            # Todos los bloqueados empatan en −∞: su orden relativo no significa nada.
            bottom_analysis.append({
                "town": item["town"], "name": item["display_name"],
                "base_rank": int(np.flatnonzero(base_order == idx)[0]) + 1,
                "blocked_by_constraint": True,
                "stays_bottom5_pct": None, "enters_top5_pct": 0.0, "wins_pct": 0.0,
                "rank_sd": None,
                "competitive_when": "Nunca: está excluido en el 100 % de los futuros por la restricción.",
            })
            continue
        town_ranks = ranks[:, idx] + 1
        enters_top5 = town_ranks <= 5
        row = {
            "town": item["town"], "name": item["display_name"],
            "base_rank": int(np.flatnonzero(base_order == idx)[0]) + 1,
            "blocked_by_constraint": bool(unsafe[idx]),
            "stays_bottom5_pct": round(float(np.mean(town_ranks > len(engine.towns) - 5)) * 100, 2),
            "enters_top5_pct": round(float(np.mean(enters_top5)) * 100, 3),
            "wins_pct": round(float(np.mean(town_ranks == 1)) * 100, 3),
            "rank_sd": round(float(np.std(town_ranks)), 2),
            "competitive_when": None,
        }
        if enters_top5.any() and not unsafe[idx]:
            taste = weights[active_indices][enters_top5].mean(axis=0)
            top_facet = int(np.argmax(taste))
            row["competitive_when"] = (
                f"En los futuros donde entra al top 5 la atención media a "
                f"{facets[top_facet]['label']} es {taste[top_facet] * 100:.0f} %."
            )
        bottom_analysis.append(row)

    mean_preference = weights[active_indices].mean(axis=0)
    mean_preference /= mean_preference.sum()
    question = _sacrifice_question(
        engine, winners, weights, active, facets, normalized_choices
    )
    trajectory = _speculative_trajectory(
        engine, matrix, relevance, mean_preference, facets
    )

    return {
        "query": intent.original,
        "mode": "strange loop multiverse",
        "risk": round(float(risk), 2),
        "total_futures": int(futures),
        "surviving_futures": int(len(active_indices)),
        "collapsed_futures": int(futures - len(active_indices)),
        "decision_entropy": round(_entropy(active_winners, len(engine.towns)) * 100.0, 2),
        "facets": [
            {
                "label": facet["label"],
                "weight": round(float(mean_preference[idx] * 100.0), 1),
            }
            for idx, facet in enumerate(facets)
        ],
        "choices": normalized_choices,
        "destinations": destinations,
        "question": question,
        "monte_carlo": monte_carlo,
        # Proporción de futuros ganados por TODOS los destinos (no solo las 5 cuencas visibles).
        "all_shares": {
            engine.towns[i]: round(100.0 * int(c) / n_active, 3)
            for i, c in enumerate(counts) if c
        },
        "bottom_in_multiverse": bottom_analysis,
        "model": {
            "weights": "w ~ Dirichlet(α·1_K), α = (28 − 24ρ)/K: menor concentración con más riesgo",
            "utility": "U(d) = 0.16·r(d) + 0.84·Σ_f w_f M[f,d] + ε, ε ~ Gumbel(0, 0.015 + 0.065ρ)",
            "choice": "destino = argmax_d U(d) (equivale a un logit multinomial dado w)",
            "constraints": "U(d) = −∞ si d tiene evidencia explícita de incumplir una exclusión",
            "entropy": "H = −Σ p_d ln p_d / ln 40 (0 = un destino seguro, 1 = uniforme)",
            "information_gain": "IG = H(antes) − Σ_b (n_b/n)·H(rama b)",
            "collapse": "responder conserva solo los futuros con w_keep ≥ w_sacrifice",
        },
        "trajectory": trajectory,
        "excluded_concepts": intent.negative,
        "disclaimer": (
            "Las trayectorias son simulaciones exploratorias de preferencias, no predicciones "
            "psicológicas ni causales."
        ),
        "seed": seed,
    }
