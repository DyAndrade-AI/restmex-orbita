"""BM25 agregado por pueblo, sin sesgo por volumen de reseñas.

Para cada reseña r y término t (Robertson y Zaragoza, 2009):

    sat(r, t) = tf(r,t)·(k1+1) / (tf(r,t) + k1·(1 − b + b·|r|/avgdl))
    idf(t)    = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

El puntaje de un pueblo d para la consulta q es el BM25 MEDIO de sus reseñas:

    S(d, q) = Σ_{t∈q} idf(t) · (1/n_d) Σ_{r∈d} sat(r, t)

Al promediar (en vez de sumar o concatenar reseñas) un pueblo con 45,000 reseñas no gana por
volumen. La media por término es lineal, así que se precomputa una matriz exacta 40 × V.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp


def build_bm25(
    counts: sp.csr_matrix, town_ids: np.ndarray, n_towns: int, k1: float = 1.2, b: float = 0.75
) -> dict:
    counts = counts.tocsr().astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    avgdl = float(lengths.mean()) or 1.0
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
    n_docs = counts.shape[0]
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    sat = counts.copy()
    row_norm = np.repeat(k1 * (1 - b + b * lengths / avgdl), np.diff(sat.indptr)).astype(np.float32)
    sat.data = sat.data * (k1 + 1) / (sat.data + row_norm)
    per_town = np.bincount(town_ids, minlength=n_towns).astype(np.float32)
    membership = sp.csr_matrix(
        (1.0 / per_town[town_ids], (town_ids, np.arange(n_docs))), shape=(n_towns, n_docs)
    )
    mean_sat = (membership @ sat).astype(np.float32).tocsc()
    return {"mean_sat": mean_sat, "idf": idf, "k1": k1, "b": b}


def save_bm25(path: Path, model: dict) -> None:
    sp.save_npz(path, model["mean_sat"].tocsr())
    np.save(Path(path).with_suffix(".idf.npy"), model["idf"])


class BM25Profiles:
    def __init__(self, path: Path, vocabulary: dict):
        self.mean_sat = sp.load_npz(path).tocsc()
        self.idf = np.load(Path(path).with_suffix(".idf.npy"))
        self.vocabulary = vocabulary

    def score(self, terms: Iterable[str]) -> np.ndarray:
        columns = sorted({self.vocabulary[t] for t in terms if t in self.vocabulary})
        if not columns:
            return np.zeros(self.mean_sat.shape[0], dtype=np.float32)
        block = self.mean_sat[:, columns].toarray()
        return (block * self.idf[columns][None, :]).sum(axis=1).astype(np.float32)
