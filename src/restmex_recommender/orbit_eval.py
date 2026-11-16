"""Validación de las dos piezas audaces sin juicios humanos.

1. Usuarios simulados para Strange Loop: cada usuario tiene un peso de gustos oculto w*; su destino
   ideal es argmax 0.16·r + 0.84·w*·M (la misma utilidad del modelo, sin ruido). Responde con
   honestidad a las preguntas de sacrificio y medimos cuánto acierta el modo del multiverso tras
   0..T preguntas, comparando la pregunta de máxima información contra una pregunta al azar.
   Dos poblaciones: "prior" (w* sale del mismo Dirichlet que el modelo, caso optimista) y
   "concentrada" (gustos dominados por una faceta, α = 0.35: el modelo está mal especificado).
2. Sensibilidad de los pesos de rol: cada coeficiente de ROLE_WEIGHTS se multiplica por un factor en
   [1 − s, 1 + s] y se mide con qué frecuencia cada rol conserva el mismo destino.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

from .futures import (
    ROLE_COPY,
    ROLE_WEIGHTS,
    _entropy,
    _sacrifice_question,
    constellation_inputs,
    multiverse_core,
    select_roles,
)

if TYPE_CHECKING:
    from .engine import Recommender

POPULATIONS = {"prior": None, "concentrada": 0.35}


def _mode_top(winners: np.ndarray, towns: int, k: int = 5) -> np.ndarray:
    counts = np.bincount(winners, minlength=towns)
    return np.argsort(-counts, kind="stable")[:k]


def simulate_users(
    engine: "Recommender",
    query: str,
    risk: float = 0.55,
    users: int = 150,
    futures: int = 20_000,
    max_questions: int = 3,
    population: str = "prior",
    seed: int = 0,
) -> Optional[dict]:
    """Curvas de acierto por número de preguntas para una consulta; None si tiene < 2 facetas."""
    core = multiverse_core(engine, query, risk, futures)
    facets, matrix, relevance = core["facets"], core["matrix"], core["relevance"]
    weights, winners, unsafe = core["weights"], core["winners"], core["unsafe"]
    k, towns = len(facets), len(engine.towns)
    if k < 2:
        return None
    labels = [f["label"] for f in facets]
    rng = np.random.default_rng(seed)
    alpha = POPULATIONS[population] or (28.0 - 24.0 * risk) / k
    hidden = rng.dirichlet(np.full(k, alpha), size=users)
    utility = 0.16 * relevance[None, :] + 0.84 * (hidden @ matrix)
    utility[:, unsafe] = -np.inf
    ideal = np.argmax(utility, axis=1)
    anchor = int(np.argmax(relevance))

    curves = {}
    for policy in ("informativa", "azar"):
        hit1 = np.zeros((users, max_questions + 1))
        hit5 = np.zeros((users, max_questions + 1))
        entropy = np.zeros((users, max_questions + 1))
        asked = np.zeros(users)
        for u in range(users):
            active = np.ones(len(winners), dtype=bool)
            choices: List[dict] = []
            for t in range(max_questions + 1):
                if t:
                    pair = None
                    if policy == "informativa":
                        q = _sacrifice_question(engine, winners, weights, active, facets, choices)
                        if q:
                            pair = (labels.index(q["branches"][0]["keep"]),
                                    labels.index(q["branches"][0]["sacrifice"]))
                    else:
                        used = {frozenset((c["keep"], c["sacrifice"])) for c in choices}
                        free = [(a, b) for a in range(k) for b in range(a + 1, k)
                                if frozenset((labels[a], labels[b])) not in used]
                        if free:
                            pair = free[rng.integers(len(free))]
                    if pair is not None:
                        a, b = pair
                        keep, drop = (a, b) if hidden[u, a] >= hidden[u, b] else (b, a)
                        narrowed = active & (weights[:, keep] >= weights[:, drop])
                        if narrowed.sum() >= 50:  # mismo mínimo que simulate_multiverse
                            active = narrowed
                            choices.append({"keep": labels[keep], "sacrifice": labels[drop]})
                            asked[u] += 1
                top = _mode_top(winners[active], towns)
                hit1[u, t] = top[0] == ideal[u]
                hit5[u, t] = ideal[u] in top
                entropy[u, t] = _entropy(winners[active], towns)
        curves[policy] = {
            "hit_at_1": hit1.mean(axis=0).round(4).tolist(),
            "hit_at_5": hit5.mean(axis=0).round(4).tolist(),
            "entropy": entropy.mean(axis=0).round(4).tolist(),
            "mean_questions_asked": round(float(asked.mean()), 3),
        }
    return {
        "query": query,
        "facets": labels,
        "population": population,
        "users": users,
        "anchor_hit_at_1": round(float(np.mean(ideal == anchor)), 4),
        "distinct_ideals": int(len(set(ideal.tolist()))),
        "curves": curves,
    }


def _bootstrap_mean_ci(values: Sequence[float], seed: int = 0, draws: int = 5_000) -> List[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def summarize_users(rows: Sequence[dict]) -> dict:
    """Promedia por consulta (cada consulta pesa igual) y da IC95 bootstrap de la ganancia."""
    t_max = len(rows[0]["curves"]["informativa"]["hit_at_1"])
    out: Dict[str, object] = {"queries": len(rows)}
    for policy in ("informativa", "azar"):
        out[policy] = {
            metric: np.mean([r["curves"][policy][metric] for r in rows], axis=0).round(4).tolist()
            for metric in ("hit_at_1", "hit_at_5", "entropy")
        }
    out["anchor_hit_at_1"] = round(float(np.mean([r["anchor_hit_at_1"] for r in rows])), 4)
    gains = {}
    for t in range(1, t_max):
        diff = [r["curves"]["informativa"]["hit_at_1"][t] - r["curves"]["azar"]["hit_at_1"][t]
                for r in rows]
        lift = [r["curves"]["informativa"]["hit_at_1"][t] - r["curves"]["informativa"]["hit_at_1"][0]
                for r in rows]
        gains[str(t)] = {
            "informativa_menos_azar": round(float(np.mean(diff)), 4),
            "ci95": _bootstrap_mean_ci(diff, seed=t),
            "mejora_vs_sin_preguntas": round(float(np.mean(lift)), 4),
            "ci95_mejora": _bootstrap_mean_ci(lift, seed=100 + t),
        }
    out["hit_at_1_gain"] = gains
    return out


def _perturbed(weights: dict, factors: Dict[tuple, float]) -> dict:
    w = copy.deepcopy(weights)
    for (role, coef), factor in factors.items():
        w[role][coef] *= factor
    return w


def role_sensitivity(
    engine: "Recommender",
    queries: Sequence[str],
    risk: float = 0.55,
    spread: float = 0.5,
    draws: int = 200,
    seed: int = 0,
) -> dict:
    """% de veces que cada rol conserva su destino cuando los pesos de diseño se perturban."""
    coefficients = [(role, coef) for role, table in ROLE_WEIGHTS.items() for coef in table]
    roles = [name for name, _ in ROLE_COPY]
    rng = np.random.default_rng(seed)
    same = np.zeros((len(queries), draws, len(roles)))
    jaccard = np.zeros((len(queries), draws))
    one_at_a_time = {c: [] for c in coefficients}
    for qi, query in enumerate(queries):
        x = constellation_inputs(engine, query, risk)
        base = select_roles(x, risk)
        for c in coefficients:
            for factor in (1 - spread, 1 + spread):
                sel = select_roles(x, risk, _perturbed(ROLE_WEIGHTS, {c: factor}))
                one_at_a_time[c].append(np.mean([a == b for a, b in zip(sel[1:], base[1:])]))
        for d in range(draws):
            factors = {c: float(rng.uniform(1 - spread, 1 + spread)) for c in coefficients}
            sel = select_roles(x, risk, _perturbed(ROLE_WEIGHTS, factors))
            same[qi, d] = [a == b for a, b in zip(sel, base)]
            jaccard[qi, d] = len(set(sel) & set(base)) / len(set(sel) | set(base))
    per_coef = {f"{r}.{c}": round(float(np.mean(v)) * 100, 1) for (r, c), v in one_at_a_time.items()}
    return {
        "queries": len(queries),
        "risk": risk,
        "spread": spread,
        "draws_per_query": draws,
        "role_unchanged_pct": {
            role: round(float(same[:, :, i].mean()) * 100, 1) for i, role in enumerate(roles)
        },
        "portfolio_jaccard_mean": round(float(jaccard.mean()), 4),
        "portfolio_identical_pct": round(float((jaccard == 1).mean()) * 100, 1),
        "one_at_a_time_unchanged_pct": dict(sorted(per_coef.items(), key=lambda kv: kv[1])),
    }
