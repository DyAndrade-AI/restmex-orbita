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
    normalize_text,
    parse_query,
    positive_segments,
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


# Conceptos de faceta. Cada uno es UN átomo de gusto aunque la consulta use varios sinónimos
# ("niños" + "familiares" → familia). Los conceptos léxicos reutilizan CONCEPT_EXPANSIONS como
# sonda; los disparadores extra de aquí solo afectan a Órbita y Strange Loop (el ranking 5 + 5
# no cambia). Las frases de varias palabras ("aire libre") se reconocen antes que los tokens.
_LEXICAL_LABELS = {
    "playa": "playa", "cenote": "cenotes", "pirámide": "arqueología", "vino": "vino",
    "bosque": "bosque y montaña", "colonial": "arquitectura colonial", "cascada": "cascadas",
    "termal": "aguas termales", "desierto": "desierto", "comida": "gastronomía",
}
_EXTRA_TRIGGERS = {
    "pirámide": ("ruinas", "maya", "mayas", "zona arqueologica"),
    "comida": ("comer",),
}
FACET_CONCEPTS = tuple(
    (_LEXICAL_LABELS[triggers[0]], triggers + _EXTRA_TRIGGERS.get(triggers[0], ()), probe)
    for triggers, probe in CONCEPT_EXPANSIONS.items()
) + (
    ("romántico", ("romantico", "romantica", "pareja", "aniversario", "enamorados", "novia",
                   "novio"), "romantico pareja aniversario cena atardecer"),
    ("familia", ("familia", "familiar", "ninos", "nino", "ninas", "hijos", "infantil"),
     "familia familiar ninos juegos parque infantil"),
    # Sin "noche": en viajes suele ser duración ("tres noches"), no un gusto.
    ("vida nocturna", ("vida nocturna", "nocturna", "nocturno", "bar", "antro", "fiesta"),
     "vida nocturna bar bares antro fiesta musica noche"),
    ("descanso", ("tranquilo", "tranquila", "tranquilidad", "descansar", "descanso", "relajarse",
                  "relajante", "relajacion", "desconectar", "paz"),
     "tranquilo tranquilidad descanso relajarse paz silencio"),
    ("vistas", ("fotografia", "fotos", "foto", "mirador", "vistas", "panoramica"),
     "fotografia fotos mirador vista vistas panoramica"),
    ("aventura", ("aventura", "aire libre", "extrema", "extremo", "tirolesa", "rappel",
                  "cuatrimoto", "escalada", "canopy"),
     "aventura tirolesa rappel cuatrimoto escalada extremo"),
    ("lago y río", ("lago", "laguna", "lancha", "rio"), "lago laguna lancha rio paseo agua"),
    ("café", ("cafe", "cafetal"), "cafe cafetal cafetales finca"),
    ("artesanía", ("artesania", "artesanal", "alfareria", "textiles"),
     "artesania artesanias artesanal alfareria textiles"),
)
# Un término sin concepto solo es faceta si DISCRIMINA entre pueblos: la entropía normalizada
# de su peso en los 40 perfiles debe ser < 0.90. Medido sobre el corpus, no sobre la evaluación:
# "ver", "rico", "aire", "libre", "tradicional" ≥ 0.92 (aparecen igual en todos); "viñedos" 0.26,
# "vegetación" 0.81, "naturaleza" 0.82. Un término uniforme no puede ser un eje de decisión.
FACET_MAX_SPREAD = 0.90


def _forms(token: str) -> set:
    """Variantes de un token canónico: sin clítico ("relajarme") y sin plural ("viñedos")."""
    forms = {token}
    for clitic in ("me", "te", "se", "nos"):
        stem = token[: -len(clitic)]
        if token.endswith(clitic) and stem.endswith(("ar", "er", "ir")):
            forms.add(stem)
    for form in list(forms):
        if form.endswith("es") and len(form) > 4:
            forms.add(form[:-2])
        if form.endswith("s") and len(form) > 3:
            forms.add(form[:-1])
    return forms


_CONCEPT_LABELS = {label for label, _, _ in FACET_CONCEPTS}
_CONCEPT_INDEX = [
    (label, {f for t in triggers if " " not in t for f in _forms(canonical_text(t))},
     tuple(canonical_text(t) for t in triggers if " " in t), probe)
    for label, triggers, probe in FACET_CONCEPTS
]


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


def _token_spread(engine: "Recommender", token: str) -> float:
    """Entropía normalizada (0–1) del peso del término entre los perfiles de pueblo."""
    rows = [engine.profile_index[(town, "ALL")] for town in engine.towns]
    column = engine.lexical_profiles[rows][:, engine.vectorizer.vocabulary_[token]].toarray().ravel()
    total = float(column.sum())
    if total <= 0:
        return 1.0
    share = column[column > 0] / total
    return float(-(share * np.log(share)).sum() / np.log(len(rows)))


def _extract_facets(engine: "Recommender", query: str, limit: int = 5) -> List[dict]:
    """Átomos de gusto: un concepto por núcleo de cada segmento coordinado de la consulta.

    1. La consulta se corta en segmentos (comas, "y", "pero"); la exclusión ya no está.
    2. En cada segmento, los conceptos de dominio son núcleos; el resto del segmento son
       modificadores que se suman a la sonda del núcleo ("gastronomía tradicional").
    3. Un segmento sin concepto da como mucho UNA faceta: su término más raro entre los que
       discriminan entre pueblos (FACET_MAX_SPREAD).
    4. Nada que coincida con la exclusión puede ser faceta positiva ("playa … sin playa")."""
    intent = parse_query(query)
    excluded = {f for t in canonical_text(intent.negative).split() for f in _forms(t)}
    vocabulary, idf = engine.vectorizer.vocabulary_, engine.vectorizer.idf_
    spelled = {canonical_text(w): w for w in normalize_text(query).split()}  # "vegetación"
    facets: List[dict] = []
    for segment in positive_segments(query):
        padded = f" {segment} "
        tokens = [t for t in segment.split() if not _forms(t) & excluded]
        consumed, heads = set(), []
        for label, singles, phrases, probe in _CONCEPT_INDEX:
            hits = [t for t in tokens if _forms(t) & singles]
            hits += [w for p in phrases if f" {p} " in padded for w in p.split()]
            consumed.update(hits)  # un concepto excluido ("sin mar") también se lleva "playa"
            if hits and not singles & excluded:
                heads.append((label, probe))
        rest = [
            t for t in tokens
            if t not in consumed and t in vocabulary and len(t) >= 3
            and t not in SPANISH_STOP_WORDS and t not in QUERY_FILLER
        ]
        if heads:
            label, probe = heads[0]
            heads[0] = (label, " ".join([probe, *rest]))
        else:
            rest = [t for t in rest if _token_spread(engine, t) < FACET_MAX_SPREAD]
            if rest:
                best = max(rest, key=lambda t: (idf[vocabulary[t]], t))
                heads = [(spelled.get(best, best), " ".join(rest))]
        for label, probe in heads:
            known = next((f for f in facets if f["label"] == label), None)
            if known:
                known["probe"] = f"{known['probe']} {probe}"
            else:
                source = "concepto turístico" if label in _CONCEPT_LABELS else "señal del usuario"
                facets.append({"label": label, "probe": probe, "source": source})

    if not facets:
        remaining = [
            t for t in canonical_text(intent.positive).split()
            if not _forms(t) & excluded and t not in SPANISH_STOP_WORDS and t not in QUERY_FILLER
            and not any(_forms(t) & singles and singles & excluded for _, singles, _, _ in _CONCEPT_INDEX)
        ]
        if not remaining:
            raise ValueError("La consulta pide y excluye lo mismo; aclara qué buscas.")
        text = " ".join(remaining)
        facets.append({"label": text, "probe": text, "source": "intención completa"})
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
    eligible: np.ndarray,
    unsafe: np.ndarray,
    penalty: float = ROLE_WEIGHTS["diversidad"]["penalty"],
) -> int:
    """Mejor candidato del rol, o −1 si no queda ningún destino seguro sin repetir.

    Restricción dura (unsafe): nunca se relaja. Umbral blando de plausibilidad (eligible): solo
    se relaja si ningún destino seguro lo cumple, y la tarjeta lo declara."""
    score = role_score.astype(np.float64).copy()
    if selected:
        diversity = distances[:, selected].mean(axis=1)
        score = (1.0 - penalty * risk) * score + penalty * risk * diversity
        score[selected] = -np.inf
    score[unsafe] = -np.inf
    if (eligible & np.isfinite(score)).any():
        score[~eligible] = -np.inf
    if not np.isfinite(score).any():
        return -1
    return int(np.argmax(score))


# Utilidad declarada de Órbita y Strange Loop (Strange Loop le suma ruido Gumbel).
UTILITY_RELEVANCE = 0.16
COUNTERFACTUAL_SCOPE = (
    "U_w(d) = 0.16·r(d) + 0.84·Σ_f w_f M[f,d]; w(s) = (1−s)·uniforme + s·v. Es la utilidad de "
    "facetas de Strange Loop, no el ranking 5 + 5: r no se descompone en facetas."
)


def _utility_gap(matrix: np.ndarray, relevance: np.ndarray, anchor: int, destination: int,
                 facet: int, shift: float) -> float:
    """U_w(s)(destino) − U_w(s)(ancla) con w(s) = (1−s)·uniforme + s·e_facet."""
    k = matrix.shape[0]
    w = np.full(k, (1.0 - shift) / k)
    w[facet] += shift
    diff = matrix[:, destination].astype(np.float64) - matrix[:, anchor]
    rel = float(relevance[destination]) - float(relevance[anchor])
    return UTILITY_RELEVANCE * rel + (1.0 - UTILITY_RELEVANCE) * float(w @ diff)


def _counterfactual(
    matrix: np.ndarray, relevance: np.ndarray, facets: Sequence[dict], anchor: int, destination: int
) -> Dict[str, object]:
    """Mínimo desplazamiento de atención s* para que el destino iguale al Ancla en U.

    Δ(s) es lineal en w, así que para cualquier s el mejor v del símplex es e_f* con
    f* = argmax_f (M[f,d] − M[f,ancla]): s* es mínimo sobre TODAS las direcciones, y si
    Δ(1) < 0 ninguna ponderación de facetas lo pone delante (imposible, no "difícil")."""
    differences = matrix[:, destination].astype(np.float64) - matrix[:, anchor]
    facet_idx = int(np.argmax(differences))
    start = _utility_gap(matrix, relevance, anchor, destination, facet_idx, 0.0)
    end = _utility_gap(matrix, relevance, anchor, destination, facet_idx, 1.0)
    if start >= 0:
        shift_percent = 0
    elif end < 0:
        shift_percent = None
    else:
        shift_percent = int(np.clip(np.ceil(-start / (end - start) * 100.0 - 1e-9), 1, 100))
        # Comprobación con la misma función: el % anunciado alcanza y uno menos no.
        while _utility_gap(matrix, relevance, anchor, destination, facet_idx,
                           shift_percent / 100.0) < 0:
            shift_percent += 1
    return {
        "facet": facets[facet_idx]["label"],
        "priority_shift": shift_percent,
        "advantage": round(float(differences[facet_idx]) * 100.0, 1),
        "utility_gap_at_equal_attention": round(start * 100.0, 2),
        "scope": "utilidad de facetas U (no el ranking 5 + 5)",
    }


def _hard_excluded(engine: "Recommender", scored: dict) -> np.ndarray:
    """Restricción dura: evidencia explícita de incumplir la exclusión (explain.violation_mask,
    la misma regla que el ranking 5 + 5) o pueblo nombrado en ella ("sin Tulum")."""
    named = engine._entity_mask(scored["intent"].negative) >= 1.0
    return np.asarray(scored["violations"]["mask"], dtype=bool) | named


def constellation_inputs(engine: "Recommender", query: str, risk: float) -> dict:
    """Todo lo que la selección de roles necesita, calculado una vez por consulta."""
    scored = engine._score_query(query)
    relevance = scored["scores"]
    facets = _extract_facets(engine, query)
    matrix = _facet_scores(engine, facets, scored["intent"].preferred_types)
    robust, worst, volatility = _scenario_metrics(matrix)

    profile_rows = [engine.profile_index[(town, "ALL")] for town in engine.towns]
    profiles = engine.lexical_profiles[profile_rows]
    similarities = (profiles @ profiles.T).toarray().astype(np.float32)
    distances = np.clip(1.0 - similarities, 0.0, 1.0)
    unsafe = _hard_excluded(engine, scored)
    if unsafe.all():
        raise ValueError("Las exclusiones descartan todos los destinos; relaja alguna.")
    # Con restricciones duras activas (por defecto) coincide con el top 1 del ranking 5 + 5.
    anchor = int(np.argmax(np.where(unsafe, -np.inf, relevance)))
    threshold = float(np.quantile(relevance, 0.60 - 0.35 * risk))
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
        pick = _pick(role_scores[role_idx], selected, x["distances"], risk,
                     x["eligible"], x["unsafe"], w["diversidad"]["penalty"])
        if pick < 0:  # menos roles antes que un destino que incumpla una exclusión
            break
        selected.append(pick)
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
        counterfactual = _counterfactual(matrix, relevance, facets, anchor, destination_idx)
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
                # El umbral blando solo se relaja si ningún destino seguro lo cumplía.
                "eligibility": "estricta" if x["eligible"][destination_idx] else "umbral relajado",
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
            cf = _counterfactual(matrix, relevance, facets, anchor, idx)
            if cf["priority_shift"] is None:
                verdict = ("Ninguna ponderación de tus facetas lo iguala al Ancla en la utilidad "
                           "de facetas: ni con toda la atención en su mejor faceta.")
            elif cf["priority_shift"] == 0:
                verdict = (f"Con atención repartida por igual ya iguala al Ancla en la utilidad de "
                           f"facetas; el ranking 5 + 5 lo deja abajo porque pesa la consulta completa.")
            else:
                verdict = (f"Con {cf['priority_shift']} % de la atención desplazada hacia "
                           f"{cf['facet']} igualaría al Ancla en la utilidad de facetas "
                           f"(no cambia su puesto en el ranking 5 + 5).")
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
            "roles_filled": len(selected),
            "counterfactual_scope": COUNTERFACTUAL_SCOPE,
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
    facets = _extract_facets(engine, query)
    matrix = _facet_scores(engine, facets, intent.preferred_types)

    seed_material = f"{intent.original}|{risk:.4f}|strange-loop-v1".encode("utf-8")
    if seed is None:
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    concentration = 28.0 - 24.0 * risk
    alpha = np.full(len(facets), concentration / len(facets), dtype=np.float64)
    weights = generator.dirichlet(alpha, size=futures).astype(np.float32)

    possible_scores = (UTILITY_RELEVANCE * relevance[None, :]
                       + (1.0 - UTILITY_RELEVANCE) * (weights @ matrix))
    # Cada línea temporal también incluye variación no observada de la experiencia. El ruido
    # Gumbel convierte la afinidad en una elección probabilística sin perder reproducibilidad.
    experience_scale = 0.015 + 0.065 * risk
    possible_scores += generator.gumbel(
        loc=0.0,
        scale=experience_scale,
        size=possible_scores.shape,
    ).astype(np.float32)
    unsafe = _hard_excluded(engine, scored)
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
        counterfactual = _counterfactual(matrix, relevance, facets, anchor, int(destination))
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
