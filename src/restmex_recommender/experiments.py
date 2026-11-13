"""Comparación de señales de recuperación con selección en calibración y reporte en prueba.

Protocolo (sin consultas escritas a mano):
1. Holdout autosupervisado: reseñas reservadas fuera del índice, nombres enmascarados.
2. División estratificada por pueblo: reseñas alternas a calibración y a prueba.
3. Para cada consulta se calculan una sola vez las cuatro señales escaladas a [0, 1]:
   perfil TF-IDF, evidencia por reseñas, BM25 medio por reseña y E5.
4. Cualquier sistema es una combinación convexa s(w) = Σ_i w_i·signal_i (+ ajustes de exclusión).
5. Se elige w en calibración (MRR) y se reporta en prueba contra el híbrido actual con un
   bootstrap pareado (IC 95 % de la diferencia). Solo se adopta un cambio si el IC excluye 0.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

SIGNALS = ("lexical", "evidence", "bm25", "semantic")


def signal_tensor(engine, queries: Sequence[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray,
                                                                       np.ndarray, List[int]]:
    """Devuelve (Q × S × 40 señales, Q × 40 ajustes, índice correcto, filas válidas)."""
    tensors, adjust, target, kept = [], [], [], []
    for i, (query, town) in enumerate(queries):
        try:
            scored = engine._score_query(query, all_signals=True)
        except ValueError:
            continue
        tensors.append(np.vstack([scored["signals"][name] for name in SIGNALS]))
        adjust.append(scored["adjust"])
        target.append(engine.town_to_idx[town])
        kept.append(i)
    return np.stack(tensors), np.stack(adjust), np.array(target), kept


def ranks_for(weights: np.ndarray, tensor: np.ndarray, adjust: np.ndarray,
              target: np.ndarray) -> np.ndarray:
    """Posición (1 = mejor) del destino correcto; empates resueltos como en el motor."""
    scores = np.einsum("s,qsd->qd", weights, tensor) + adjust
    n = scores.shape[1]
    idx = np.arange(n)
    ranks = np.empty(len(target), dtype=np.int32)
    for q in range(len(target)):
        order = np.lexsort((idx, -scores[q]))
        ranks[q] = int(np.flatnonzero(order == target[q])[0]) + 1
    return ranks


def metrics_from_ranks(ranks: np.ndarray, n: int = 40) -> Dict[str, float]:
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hit_at_1": float(np.mean(ranks == 1)),
        "hit_at_5": float(np.mean(ranks <= 5)),
        "ndcg_at_5": float(np.mean(np.where(ranks <= 5, 1.0 / np.log2(ranks + 1), 0.0))),
        "bottom5_rate": float(np.mean(ranks > n - 5)),
        "mean_rank": float(np.mean(ranks)),
    }


def simplex_grid(step: float, active: Sequence[bool]) -> List[np.ndarray]:
    units = int(round(1 / step))
    names = [i for i, on in enumerate(active) if on]
    grid = []
    for combo in itertools.product(range(units + 1), repeat=len(names)):
        if sum(combo) != units:
            continue
        w = np.zeros(len(active))
        for pos, value in zip(names, combo):
            w[pos] = value / units
        grid.append(w)
    return grid


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iterations: int = 5000,
                     seed: int = 2025) -> Dict[str, float]:
    """IC 95 % de mean(a − b) remuestreando consultas con reemplazo."""
    rng = np.random.default_rng(seed)
    diff = a - b
    samples = rng.integers(0, len(diff), size=(iterations, len(diff)))
    means = diff[samples].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            # Con B remuestreos la menor p resoluble es 1/B: se reporta como cota, nunca 0.
            "p_two_sided": float(max(1.0 / iterations,
                                     min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))))}


def stratified_halves(queries: Sequence[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
    seen: Dict[str, int] = {}
    calib, test = [], []
    for i, (_, town) in enumerate(queries):
        k = seen.get(town, 0)
        (calib if k % 2 == 0 else test).append(i)
        seen[town] = k + 1
    return calib, test


def named_systems(semantic: bool) -> Dict[str, np.ndarray]:
    def w(**kw):
        return np.array([kw.get(name, 0.0) for name in SIGNALS])

    systems = {
        "tfidf_perfil": w(lexical=1),
        "evidencia_reseñas": w(evidence=1),
        "bm25_medio": w(bm25=1),
        "lexico_actual (0.68·tfidf + 0.32·evid)": w(lexical=0.68, evidence=0.32),
        "bm25 + evidencia (0.68/0.32)": w(bm25=0.68, evidence=0.32),
    }
    if semantic:
        systems["e5"] = w(semantic=1)
        systems["hibrido_actual (0.38/0.17/0.45)"] = w(lexical=0.38, evidence=0.17, semantic=0.45)
    return systems


def run(engine, queries, step: float = 0.1) -> dict:
    tensor, adjust, target, kept = signal_tensor(engine, queries)
    kept_queries = [queries[i] for i in kept]
    calib_local, test_local = stratified_halves(kept_queries)
    calib, test = np.array(calib_local), np.array(test_local)
    semantic = bool(engine.semantic_enabled)
    systems = named_systems(semantic)
    reference = "hibrido_actual (0.38/0.17/0.45)" if semantic else (
        "lexico_actual (0.68·tfidf + 0.32·evid)")

    def evaluate(weights: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return ranks_for(weights, tensor[rows], adjust[rows], target[rows])

    results = {}
    for name, weights in systems.items():
        results[name] = {
            "weights": dict(zip(SIGNALS, map(float, weights))),
            "calibration": metrics_from_ranks(evaluate(weights, calib)),
            "test": metrics_from_ranks(evaluate(weights, test)),
        }

    grid = simplex_grid(step, [True, True, True, semantic])
    best_w, best_mrr = None, -math.inf
    for weights in grid:
        mrr = float(np.mean(1.0 / evaluate(weights, calib)))
        if mrr > best_mrr + 1e-12:
            best_w, best_mrr = weights, mrr
    tuned_test = evaluate(best_w, test)
    ref_test = evaluate(systems[reference], test)
    results["ajustado_en_calibracion"] = {
        "weights": dict(zip(SIGNALS, map(float, best_w))),
        "calibration": metrics_from_ranks(evaluate(best_w, calib)),
        "test": metrics_from_ranks(tuned_test),
    }
    return {
        "protocol": __doc__.strip(),
        "queries": {"calibration": int(len(calib)), "test": int(len(test))},
        "semantic": semantic,
        "reference": reference,
        "grid_size": len(grid),
        "systems": results,
        "tuned_vs_reference_test": {
            "mrr": paired_bootstrap(1.0 / tuned_test, 1.0 / ref_test),
            "hit_at_5": paired_bootstrap((tuned_test <= 5).astype(float),
                                         (ref_test <= 5).astype(float)),
        },
    }
