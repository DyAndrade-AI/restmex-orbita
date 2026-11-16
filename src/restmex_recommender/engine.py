"""Motor híbrido de recomendación, explicable y completamente local.

Semántica del contrato 5 + 5:
- "top" son los destinos con MAYOR afinidad temática con la consulta.
- "bottom" son los de MENOR afinidad temática. No son destinos peores: la calidad percibida
  (polaridad promedio con suavizado bayesiano) se reporta aparte y nunca entra al ranking.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import scipy.sparse as sp

from . import explain
from .text import (
    QUERY_FILLER,
    SPANISH_STOP_WORDS,
    canonical_text,
    display_label,
    expand_for_lexical,
    parse_query,
    phonetic_key,
)

# Pesos de las señales. Se exponen para ablaciones reproducibles en la evaluación.
# Híbrido elegido en calibración del holdout (docs/experimentos_hybrid.md): supera al anterior
# (0.38/0.17/0.45) en prueba con ΔMRR +0.051, IC95 [+0.022, +0.080]; empata con el léxico.
HYBRID_WEIGHTS = {"lexical": 0.40, "evidence": 0.20, "bm25": 0.30, "semantic": 0.10}
PREVIOUS_HYBRID_WEIGHTS = {"lexical": 0.38, "evidence": 0.17, "semantic": 0.45}
# Léxico: el ajuste (0.50/0.20/0.30 bm25) no mejora el MRR (criterio primario, IC95 incluye 0)
# aunque sí el Hit@5; se conserva hasta confirmarlo con juicios humanos (docs/metodologia.md §4).
LEXICAL_WEIGHTS = {"lexical": 0.68, "evidence": 0.32, "semantic": 0.0}
# Fuerza del prior bayesiano para la calidad: equivale a 200 reseñas con la media global.
QUALITY_PRIOR_REVIEWS = 200


def _scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def _rank_order(scores: np.ndarray) -> np.ndarray:
    """Orden descendente con desempate por índice (alfabético). El bottom es su reverso exacto,
    así top y bottom nunca se solapan cuando top_k + bottom_k <= n, aun con empates."""
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores)))


class Recommender:
    def __init__(
        self,
        artifact_dir: Path,
        model_dir: Optional[Path] = None,
        *,
        enable_semantic: bool = True,
    ):
        self.artifact_dir = Path(artifact_dir)
        manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        self.compat_warnings = self._check_versions(manifest)
        self.vectorizer = joblib.load(self.artifact_dir / "vectorizer.joblib")
        self.analyzer = self.vectorizer.build_analyzer()
        self.lexical_profiles = sp.load_npz(self.artifact_dir / "lexical_profiles.npz").tocsr()
        self.evidence_matrix = sp.load_npz(self.artifact_dir / "evidence_matrix.npz").tocsr()
        bundle = json.loads((self.artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        self.destinations = bundle["destinations"]
        for item in self.destinations:
            item["display_name"] = display_label(item["display_name"])
            item["display_region"] = display_label(item["display_region"])
        self.profile_mapping = bundle["profile_mapping"]
        self.evidence = bundle["evidence"]
        self.towns = [item["town"] for item in self.destinations]
        self.town_to_idx = {town: idx for idx, town in enumerate(self.towns)}
        self.profile_index = {
            (item["town"], item["profile_type"]): idx
            for idx, item in enumerate(self.profile_mapping)
        }
        self.evidence_by_town: Dict[str, np.ndarray] = {}
        for town in self.towns:
            self.evidence_by_town[town] = np.array(
                [idx for idx, item in enumerate(self.evidence) if item["Town"] == town], dtype=int
            )
        rates_path = self.artifact_dir / "town_term_rates.npz"
        self.term_rates = sp.load_npz(rates_path).tocsc() if rates_path.exists() else None
        self.quality = self._bayesian_quality()
        self.weights = dict(HYBRID_WEIGHTS)
        self._query_cache: Dict[str, np.ndarray] = {}
        # Restricción dura: con evidencia explícita (explain.violation_mask) el pueblo se hunde.
        self.hard_constraints = True
        self._spelling_index: Optional[Dict[str, str]] = None
        bm25_path = self.artifact_dir / "bm25.npz"
        if bm25_path.exists():
            from .bm25 import BM25Profiles

            self.bm25 = BM25Profiles(bm25_path, self.vectorizer.vocabulary_)
        else:
            self.bm25 = None

        self.semantic_embeddings = None
        self.semantic_mapping = None
        self.semantic_model = None
        self.semantic_status = "desactivado por configuración"
        embeddings_path = self.artifact_dir / "semantic_embeddings.npy"
        mapping_path = self.artifact_dir / "semantic_mapping.json"
        if os.environ.get("ORBITA_SEMANTIC", "1") == "0":
            enable_semantic = False
            self.semantic_status = "desactivado por ORBITA_SEMANTIC=0"
        if enable_semantic and not (embeddings_path.exists() and mapping_path.exists()):
            self.semantic_status = "sin artefactos semánticos (ejecuta make build-semantic)"
        elif enable_semantic and not manifest.get("semantic_available"):
            self.semantic_status = "artefactos semánticos ignorados: el manifest no los declara"
        elif enable_semantic and not self._hashes_match(manifest, embeddings_path, mapping_path):
            self.semantic_status = "artefactos semánticos ignorados: hash distinto al manifest"
        elif enable_semantic:
            self.semantic_embeddings = np.load(embeddings_path)
            self.semantic_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            try:
                from sentence_transformers import SentenceTransformer

                local_candidate = self.artifact_dir.parent / "models" / "multilingual-e5-base"
                if model_dir and Path(model_dir).exists():
                    source, local_only = str(model_dir), True
                elif local_candidate.exists():
                    source, local_only = str(local_candidate), True
                else:
                    source, local_only = manifest["semantic_model"], False
                self.semantic_model = SentenceTransformer(
                    source,
                    local_files_only=local_only,
                    tokenizer_kwargs={"fix_mistral_regex": True},
                )
                self.semantic_status = f"activo ({Path(source).name})"
            except Exception as exc:  # noqa: BLE001 - el fallback léxico es intencional
                # La recuperación léxica sigue funcionando, pero el motivo queda visible en /health.
                self.semantic_model = None
                self.semantic_status = f"fallback léxico: {type(exc).__name__}: {exc}"[:300]
        if not self.semantic_enabled:
            self.weights = dict(LEXICAL_WEIGHTS)

    @staticmethod
    def _check_versions(manifest: dict) -> List[str]:
        """Compara versiones del runtime con las que generaron los artefactos."""
        import scipy
        import sklearn

        expected = manifest.get("runtime_versions") or {"scikit-learn": manifest.get("sklearn")}
        current = {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
        }
        return [
            f"{name}: artefactos {version}, runtime {current[name]}"
            for name, version in expected.items()
            if version and name in current and current[name] != version
        ]

    @staticmethod
    def _hashes_match(manifest: dict, *paths: Path) -> bool:
        expected = manifest.get("artifacts_sha256")
        if expected is None:  # manifiestos anteriores a los hashes
            return True
        return all(
            expected.get(p.name) == hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        )

    @property
    def semantic_enabled(self) -> bool:
        return self.semantic_model is not None and self.semantic_embeddings is not None

    def _bayesian_quality(self) -> np.ndarray:
        counts = np.array([d["review_count"] for d in self.destinations], dtype=np.float64)
        means = np.array([d["rating_mean"] for d in self.destinations], dtype=np.float64)
        prior = float((counts * means).sum() / counts.sum())
        m = QUALITY_PRIOR_REVIEWS
        return (counts * means + m * prior) / (counts + m)

    def _profile_scores(
        self, query_vector: sp.csr_matrix, preferred_types: Sequence[str]
    ) -> np.ndarray:
        all_scores = (self.lexical_profiles @ query_vector.T).toarray().ravel()
        result = np.zeros(len(self.towns), dtype=np.float32)
        relevant_types = preferred_types or ("Attractive", "Hotel", "Restaurant")
        for i, town in enumerate(self.towns):
            general = all_scores[self.profile_index[(town, "ALL")]]
            type_scores = [all_scores[self.profile_index[(town, kind)]] for kind in relevant_types]
            result[i] = 0.78 * general + 0.22 * max(type_scores, default=0.0)
        return result

    def _evidence_scores(self, query_vector: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        similarities = (self.evidence_matrix @ query_vector.T).toarray().ravel()
        scores = np.zeros(len(self.towns), dtype=np.float32)
        best_indices = np.full(len(self.towns), -1, dtype=int)
        for i, town in enumerate(self.towns):
            indices = self.evidence_by_town[town]
            if len(indices):
                local = similarities[indices]
                take = min(3, len(local))
                top = np.partition(local, -take)[-take:]
                scores[i] = float(0.65 * top.max() + 0.35 * top.mean())
                # Sin coincidencia real no se adjunta evidencia: sería una justificación falsa.
                if local.max() > 0:
                    best_indices[i] = int(indices[int(np.argmax(local))])
        return scores, best_indices

    def bm25_scores(self, text: str) -> np.ndarray:
        """BM25 por pueblo (opcional): solo si existe artifacts/bm25.npz (ver experiments.py)."""
        if self.bm25 is None:
            return np.zeros(len(self.towns), dtype=np.float32)
        return self.bm25.score(self.analyzer(text))

    def _semantic_scores(self, query: str, preferred_types: Sequence[str]) -> np.ndarray:
        if not self.semantic_enabled:
            return np.zeros(len(self.towns), dtype=np.float32)
        vector = self._query_cache.get(query)
        if vector is None:
            vector = self.semantic_model.encode_query(
                ["query: " + query], normalize_embeddings=True, convert_to_numpy=True
            )[0]
            if len(self._query_cache) < 4_096:  # shortcut: sin LRU; basta para evaluación/demo
                self._query_cache[query] = vector
        raw = self.semantic_embeddings @ vector
        by_town: Dict[str, List[float]] = {town: [] for town in self.towns}
        for score, mapping in zip(raw, self.semantic_mapping):
            if preferred_types and mapping["profile_type"] not in ("ALL", *preferred_types):
                continue
            by_town[mapping["town"]].append(float(score))
        return np.array(
            [max(by_town[town], default=float(raw.min())) for town in self.towns], dtype=np.float32
        )

    def _entity_mask(self, text: str) -> np.ndarray:
        canonical = f" {canonical_text(text)} "
        boost = np.zeros(len(self.towns), dtype=np.float32)
        if not canonical.strip():
            return boost
        for i, item in enumerate(self.destinations):
            town = canonical_text(item["display_name"])
            region = canonical_text(item["display_region"])
            if f" {town} " in canonical:
                boost[i] += 1.0
            elif len(region) > 4 and f" {region} " in canonical:
                boost[i] += 0.30
        return boost

    def _matched_terms(self, query_vector: sp.csr_matrix, town: str, limit: int = 5) -> List[str]:
        row = self.lexical_profiles[self.profile_index[(town, "ALL")]]
        overlap = row.multiply(query_vector)
        if not overlap.nnz:
            return []
        names = self.vectorizer.get_feature_names_out()
        order = np.argsort(overlap.data)[::-1][:limit]
        return [str(names[overlap.indices[pos]]) for pos in order]

    def correct_spelling(self, text: str) -> Tuple[str, List[dict]]:
        """Corrige palabras fuera del vocabulario (≥5 letras) contra los unigramas del train.

        Primero busca coincidencia exacta de clave fonética; si no, la más parecida con
        difflib (ratio ≥ 0.8). Entre candidatos se elige el más frecuente (menor idf)."""
        import difflib

        vocab = self.vectorizer.vocabulary_
        if self._spelling_index is None:
            idf = self.vectorizer.idf_
            index: Dict[str, str] = {}
            for term, col in vocab.items():
                if " " in term:
                    continue
                key = phonetic_key(term)
                if key not in index or idf[col] < idf[vocab[index[key]]]:
                    index[key] = term
            self._spelling_index = index
        index = self._spelling_index
        corrections, words = [], []
        for word in canonical_text(text).split():
            if len(word) < 5 or word in vocab or word in SPANISH_STOP_WORDS:
                words.append(word)
                continue
            key = phonetic_key(word)
            match = index.get(key)
            if match is None:
                close = difflib.get_close_matches(key, index.keys(), n=1, cutoff=0.8)
                match = index[close[0]] if close else None
            if match and match != word:
                corrections.append({"from": word, "to": match})
                words.append(match)
            else:
                words.append(word)
        return " ".join(words), corrections

    def query_terms(self, text: str) -> List[str]:
        """Términos de la consulta (sin expansión) que existen en el vocabulario del train."""
        vocab = self.vectorizer.vocabulary_
        seen: List[str] = []
        for term in self.analyzer(text):
            words = term.split()
            if term in vocab and term not in seen and not set(words) & QUERY_FILLER:
                seen.append(term)
        return seen

    def term_rate_table(self, terms: Sequence[str]) -> Optional[np.ndarray]:
        """Matriz pueblos × términos con la fracción de reseñas que mencionan cada término."""
        if self.term_rates is None or not terms:
            return None
        cols = [self.vectorizer.vocabulary_[t] for t in terms]
        return self.term_rates[:, cols].toarray()

    def _corpus_check(self, idx: int, terms: List[str], rates: Optional[np.ndarray]) -> List[dict]:
        if rates is None:
            return []
        medians = np.median(rates, axis=0)
        rows = [
            {
                "term": term,
                "rate": round(float(rates[idx, j]) * 100.0, 2),
                "median_rate": round(float(medians[j]) * 100.0, 2),
            }
            for j, term in enumerate(terms)
        ]
        return sorted(rows, key=lambda r: -r["rate"])[:4]

    def recommend(self, query: str, top_k: int = 5, bottom_k: int = 5) -> dict:
        if top_k + bottom_k > len(self.towns):
            raise ValueError("La suma de resultados excede el número de destinos disponibles.")
        scored = self._score_query(query)
        intent = scored["intent"]
        query_vector = scored["query_vector"]
        best_evidence = scored["best_evidence"]
        final = scored["scores"]
        order = _rank_order(final)
        top_indices, bottom_indices = order[:top_k], order[::-1][:bottom_k]
        terms = self.query_terms(scored["lexical_text"])
        rates = self.term_rate_table(terms)
        groups = explain.term_groups(self, scored["lexical_text"], expand_concepts=False)
        group_rates = explain.group_rates(self, groups)
        violations = scored["violations"]
        position = {int(idx): pos + 1 for pos, idx in enumerate(order)}
        excluded_terms = {t for g in violations["groups"] for t in g["terms"]}
        contradictions = [g["label"] for g in groups if set(g["terms"]) & excluded_terms]
        confidence = dict(scored["confidence"])
        if contradictions:
            confidence["level"] = "baja"
            confidence["note"] = (
                "Consulta contradictoria: se pide y se excluye " + ", ".join(contradictions)
                + ". La restricción prevalece; reformula si no era la intención."
            )

        def serialize(idx: int, rank: int, least: bool = False) -> dict:
            item = self.destinations[idx]
            evidence_idx = best_evidence[idx]
            facts = explain.explain_destination(idx, groups, group_rates, violations)
            violation_idx = int(scored["negative_evidence"][idx])
            return {
                "rank": rank,
                "global_rank": position[int(idx)],
                "town": item["town"],
                "name": item["display_name"],
                "region": item["display_region"],
                "score": round(float(final[idx] * 100.0), 1),
                "relation": "menor afinidad temática" if least else "mayor afinidad temática",
                "matched_terms": self._matched_terms(query_vector, item["town"]),
                "keywords": item["keywords"]["ALL"][:6],
                # Las reseñas de evidencia son positivas: solo justifican afinidad, nunca lejanía.
                "evidence": (
                    self.evidence[evidence_idx] if (evidence_idx >= 0 and not least) else None
                ),
                "corpus_check": self._corpus_check(idx, terms, rates),
                "review_count": item["review_count"],
                "rating_mean": item["rating_mean"],
                "quality_bayes": round(float(self.quality[idx]), 3),
                "least_related": least,
                **facts,
                # Reseña real que menciona lo excluido: evidencia explícita, no inferida.
                "violation_evidence": (
                    self.evidence[violation_idx]
                    if facts["violated"] and violation_idx >= 0 else None
                ),
                "explanation": explain.narrative(facts, least),
            }

        return {
            "query": intent.original,
            "positive_query": intent.positive,
            "excluded_concepts": intent.negative,
            "preferred_types": intent.preferred_types,
            "mode": scored["mode"],
            "confidence": confidence,
            "contradictions": contradictions,
            "query_terms": terms,
            "score_note": (
                "Compatibilidad relativa 0–100 entre los 40 destinos para esta consulta; "
                "no es una probabilidad de satisfacción."
            ),
            "constraints": [
                {"constraint": f"sin {g['label']}", "towns_with_evidence": int(m)}
                for g, m in zip(violations["groups"], (
                    (violations["rates"] >= np.maximum(
                        explain.LIFT * violations["medians"], explain.MIN_VIOLATION
                    )[None, :]).sum(axis=0) if violations["groups"] else []
                ))
            ],
            "top": [serialize(idx, rank + 1) for rank, idx in enumerate(top_indices)],
            "bottom": [serialize(idx, rank + 1, True) for rank, idx in enumerate(bottom_indices)],
        }

    def _score_query(
        self, query: str, weights: Optional[Dict[str, float]] = None, all_signals: bool = False
    ) -> dict:
        """Calcula afinidades para todo el inventario sin recortar el ranking."""
        intent = parse_query(query)
        if len(intent.original.split()) < 2:
            raise ValueError("Describe al menos dos palabras sobre el viaje que buscas.")
        if not intent.positive.strip():
            raise ValueError("La consulta solo contiene exclusiones; di también qué buscas.")

        lexical_text, corrections = self.correct_spelling(intent.positive)
        raw_vector = self.vectorizer.transform([lexical_text])
        query_vector = self.vectorizer.transform([expand_for_lexical(lexical_text)])
        content_words = [
            w for w in canonical_text(lexical_text).split()
            if w not in SPANISH_STOP_WORDS and w not in QUERY_FILLER
        ]
        known = [w for w in content_words if w in self.vectorizer.vocabulary_]
        coverage = len(known) / len(content_words) if content_words else 0.0

        w = dict(weights or self.weights)
        if w.get("semantic", 0) > 0 and not self.semantic_enabled:
            raise ValueError("Se pidió la señal semántica pero el modelo no está cargado.")
        if query_vector.nnz == 0 and w.get("semantic", 0) == 0:
            raise ValueError(
                "Ninguna palabra de la consulta aparece en las reseñas del train; "
                "el ranking sería arbitrario. Describe actividades, paisajes o comida."
            )

        lexical = self._profile_scores(query_vector, intent.preferred_types)
        evidence, best_evidence = self._evidence_scores(query_vector)
        want_semantic = w.get("semantic", 0) > 0 or (all_signals and self.semantic_enabled)
        zeros = np.zeros(len(self.towns), dtype=np.float32)
        signals = {
            "lexical": _scale(lexical),
            "evidence": _scale(evidence),
            "semantic": (
                _scale(self._semantic_scores(intent.positive, intent.preferred_types))
                if want_semantic else zeros
            ),
            "bm25": (
                _scale(self.bm25_scores(lexical_text))
                if (w.get("bm25", 0) > 0 or all_signals) else zeros
            ),
        }
        combined = sum(w.get(name, 0) * values for name, values in signals.items())
        base = combined.copy()
        mode = "híbrido semántico + léxico" if w.get("semantic", 0) > 0 else "léxico local"

        negative_best = np.full(len(self.towns), -1, dtype=int)
        violations = explain.violation_mask(self, intent.negative)
        if intent.negative:
            negative_vector = self.vectorizer.transform([expand_for_lexical(intent.negative)])
            if negative_vector.nnz:
                negative_lexical = self._profile_scores(negative_vector, intent.preferred_types)
                negative_evidence, negative_best = self._evidence_scores(negative_vector)
                penalty = 0.65 * _scale(negative_lexical) + 0.35 * _scale(negative_evidence)
                combined = combined - 0.28 * penalty
            # Un pueblo nombrado como exclusión ("sin Tulum") se hunde explícitamente.
            combined = combined - 1.0 * self._entity_mask(intent.negative)
            if self.hard_constraints:
                combined = combined - 1.0 * violations["mask"].astype(np.float32)

        combined = combined + 0.35 * self._entity_mask(intent.positive)
        spread = float(combined.max() - combined.min())
        if spread < 1e-6:
            raise ValueError("La consulta no distingue entre destinos; agrega más detalles.")

        if raw_vector.nnz == 0 and coverage == 0:
            level = "baja"
            note = "Sin términos del corpus: solo responde la señal semántica."
        elif coverage < 0.5:
            level = "media"
            note = "Parte de la consulta no aparece en las reseñas del train."
        else:
            level = "alta"
            note = "La consulta está bien cubierta por el vocabulario del train."
        final = _scale(combined)
        return {
            "signals": signals,
            "lexical_text": lexical_text,
            # Ajustes no lineales en los pesos (exclusiones, entidades) para experimentos.
            "adjust": combined - base,
            "intent": intent,
            "query_vector": query_vector,
            "best_evidence": best_evidence,
            "scores": final,
            "mode": mode,
            "violations": violations,
            "negative_evidence": negative_best,
            "confidence": {
                "level": level,
                "vocab_coverage": round(coverage, 3),
                "unknown_terms": [x for x in content_words if x not in known][:8],
                "note": note,
                "spelling_corrections": corrections,
            },
        }

    def recommend_futures(self, query: str, risk: float = 0.55, count: int = 5) -> dict:
        """Crea una cartera diversa y contrafactual de futuros de viaje."""
        from .futures import build_future_constellation

        return build_future_constellation(self, query=query, risk=risk, count=count)

    def explore_multiverse(
        self,
        query: str,
        risk: float = 0.55,
        futures: int = 50_000,
        choices: Optional[List[dict]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Simula versiones de preferencias y permite colapsarlas con decisiones."""
        from .futures import simulate_multiverse

        return simulate_multiverse(
            self,
            query=query,
            risk=risk,
            futures=futures,
            choices=choices or [],
            seed=seed,
        )
