"""Construcción reproducible de los índices usando solo RestMex train."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .bm25 import build_bm25, save_bm25
from .text import SPANISH_STOP_WORDS, normalize_text

EXPECTED_COLUMNS = {"Title", "Review", "Polarity", "Town", "Region", "Type"}
PROFILE_TYPES = ("ALL", "Attractive", "Hotel", "Restaurant")


def runtime_versions() -> Dict[str, str]:
    import joblib as _joblib
    import scipy

    return {
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "joblib": _joblib.__version__,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_centroids(
    matrix: sp.csr_matrix,
    group_ids: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
) -> sp.csr_matrix:
    rows = np.arange(matrix.shape[0])
    assignment = sp.csr_matrix((weights, (group_ids, rows)), shape=(n_groups, matrix.shape[0]))
    centroids = assignment @ matrix
    denominators = np.asarray(assignment.sum(axis=1)).ravel()
    denominators[denominators == 0] = 1.0
    centroids = sp.diags(1.0 / denominators) @ centroids
    return normalize(centroids, norm="l2", axis=1).tocsr()


def _top_terms(row: sp.csr_matrix, feature_names: np.ndarray, limit: int = 28) -> List[str]:
    if row.nnz == 0:
        return []
    order = np.argsort(row.data)[::-1]
    terms: List[str] = []
    seen = set()
    for pos in order:
        term = str(feature_names[row.indices[pos]])
        # Prefer useful phrases and avoid returning a unigram already represented in a phrase.
        key = term.replace(" ", "_")
        if key not in seen:
            terms.append(term)
            seen.add(key)
        if len(terms) >= limit:
            break
    return terms


def _sample_evidence(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    candidates = df[(df["Polarity"] >= 4) & df["text"].str.len().between(35, 520)].copy()
    candidates = candidates.drop_duplicates(["Town", "Type", "text"])
    parts = []
    for (_, _), group in candidates.groupby(["Town", "Type"], sort=True):
        n = min(per_group, len(group))
        # A stable mixture: strongest ratings plus a random sample to preserve niche activities.
        high = group[group["Polarity"] == 5]
        chosen = high.sample(min(n // 2, len(high)), random_state=seed)
        remaining = group.drop(chosen.index)
        if len(chosen) < n:
            chosen = pd.concat(
                [chosen, remaining.sample(min(n - len(chosen), len(remaining)), random_state=seed + 1)]
            )
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True).sort_values(["Town", "Type"]).reset_index(drop=True)


def _semantic_documents(metadata: List[dict]) -> Tuple[List[str], List[dict]]:
    documents: List[str] = []
    mapping: List[dict] = []
    labels = {
        "ALL": "experiencia general",
        "Attractive": "atracciones y actividades",
        "Hotel": "hospedaje",
        "Restaurant": "gastronomía",
    }
    for destination in metadata:
        for profile_type in PROFILE_TYPES:
            terms = destination["keywords"].get(profile_type, [])
            if not terms:
                continue
            text = (
                f"Pueblo Mágico: {destination['display_name']}. "
                f"Estado o región: {destination['display_region']}. "
                f"Perfil de {labels[profile_type]} basado en reseñas de visitantes: "
                + ", ".join(terms)
                + "."
            )
            documents.append(text)
            mapping.append({"town": destination["town"], "profile_type": profile_type})
    return documents, mapping


def build_index(
    data_path: Path,
    output_dir: Path,
    *,
    with_semantic: bool = False,
    model_name: str = "intfloat/multilingual-e5-base",
    model_dir: Optional[Path] = None,
    max_features: int = 60_000,
    evidence_per_group: int = 40,
    seed: int = 2025,
    exclude_rows: Optional[np.ndarray] = None,
) -> dict:
    """Construye los índices. ``exclude_rows`` (máscara booleana) reserva filas como holdout:
    no participan en vocabulario, perfiles, evidencia ni embeddings, evitando fuga."""
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df.empty:
        raise ValueError("El archivo de entrenamiento está vacío")

    total_rows = int(len(df))
    if exclude_rows is not None:
        exclude_rows = np.asarray(exclude_rows, dtype=bool)
        if exclude_rows.shape != (len(df),):
            raise ValueError("exclude_rows debe tener una entrada por fila del CSV")
        df = df.loc[~exclude_rows]
    df = df.copy().reset_index(drop=True)
    df["Title"] = df["Title"].fillna("")
    df["Review"] = df["Review"].fillna("")
    df["text"] = (df["Title"].astype(str) + " " + df["Review"].astype(str)).map(normalize_text)
    df["Polarity"] = pd.to_numeric(df["Polarity"], errors="raise").clip(1, 5)
    towns = sorted(df["Town"].unique().tolist())
    town_to_id = {town: idx for idx, town in enumerate(towns)}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=4,
        max_df=0.92,
        max_features=max_features,
        stop_words=sorted(SPANISH_STOP_WORDS),
        dtype=np.float32,
        norm="l2",
    )
    review_matrix = vectorizer.fit_transform(df["text"])
    # Positive reviews weigh more, but low ratings remain represented.
    quality_weights = (0.35 + 0.65 * (df["Polarity"].to_numpy(np.float32) - 1.0) / 4.0)
    town_ids = df["Town"].map(town_to_id).to_numpy()

    profile_rows: List[sp.csr_matrix] = []
    profile_mapping: List[dict] = []
    global_centroids = _weighted_centroids(
        review_matrix, town_ids, quality_weights, len(towns)
    )
    for town_idx, town in enumerate(towns):
        profile_rows.append(global_centroids[town_idx])
        profile_mapping.append({"town": town, "profile_type": "ALL"})
        for profile_type in PROFILE_TYPES[1:]:
            mask = (df["Town"].to_numpy() == town) & (df["Type"].to_numpy() == profile_type)
            if not mask.any():
                profile_rows.append(sp.csr_matrix((1, review_matrix.shape[1]), dtype=np.float32))
            else:
                local = review_matrix[mask]
                local_weights = quality_weights[mask]
                centroid = local.multiply(local_weights[:, None]).sum(axis=0)
                centroid = sp.csr_matrix(centroid / max(float(local_weights.sum()), 1.0))
                profile_rows.append(normalize(centroid, norm="l2", axis=1))
            profile_mapping.append({"town": town, "profile_type": profile_type})
    lexical_profiles = sp.vstack(profile_rows, format="csr")

    # Fracción de reseñas de cada pueblo que contienen cada término: verificación auditable.
    presence = review_matrix.copy()
    presence.data[:] = 1.0
    counts = np.bincount(town_ids, minlength=len(towns)).astype(np.float32)
    membership = sp.csr_matrix(
        (1.0 / counts[town_ids], (town_ids, np.arange(len(df)))), shape=(len(towns), len(df))
    )
    town_term_rates = (membership @ presence).astype(np.float32).tocsr()
    town_term_rates.data[town_term_rates.data < 1e-4] = 0.0
    town_term_rates.eliminate_zeros()

    feature_names = vectorizer.get_feature_names_out()
    metadata: List[dict] = []
    for town in towns:
        town_df = df[df["Town"] == town]
        row_by_type: Dict[str, sp.csr_matrix] = {}
        for idx, mapping in enumerate(profile_mapping):
            if mapping["town"] == town:
                row_by_type[mapping["profile_type"]] = lexical_profiles[idx]
        region = str(town_df["Region"].mode().iloc[0])
        metadata.append(
            {
                "town": town,
                "display_name": town.replace("_", " "),
                "region": region,
                "display_region": region.replace("_", " "),
                "review_count": int(len(town_df)),
                "rating_mean": round(float(town_df["Polarity"].mean()), 3),
                "type_counts": {
                    kind: int((town_df["Type"] == kind).sum()) for kind in PROFILE_TYPES[1:]
                },
                "keywords": {
                    kind: _top_terms(row_by_type[kind], feature_names)
                    for kind in PROFILE_TYPES
                },
            }
        )

    evidence = _sample_evidence(df, evidence_per_group, seed)
    evidence_matrix = vectorizer.transform(evidence["text"])
    evidence_records = evidence[
        ["Town", "Region", "Type", "Polarity", "Title", "Review"]
    ].to_dict(orient="records")

    joblib.dump(vectorizer, output_dir / "vectorizer.joblib", compress=3)
    sp.save_npz(output_dir / "lexical_profiles.npz", lexical_profiles)
    sp.save_npz(output_dir / "evidence_matrix.npz", evidence_matrix)
    sp.save_npz(output_dir / "town_term_rates.npz", town_term_rates)
    counts = CountVectorizer(
        analyzer=vectorizer.build_analyzer(), vocabulary=vectorizer.vocabulary_, dtype=np.float32
    ).transform(df["text"])
    save_bm25(output_dir / "bm25.npz", build_bm25(counts, town_ids, len(towns)))
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "destinations": metadata,
                "profile_mapping": profile_mapping,
                "evidence": evidence_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    semantic_available = False
    semantic_documents, semantic_mapping = _semantic_documents(metadata)
    if with_semantic:
        from sentence_transformers import SentenceTransformer

        load_from = str(model_dir) if model_dir and model_dir.exists() else model_name
        model = SentenceTransformer(
            load_from,
            local_files_only=bool(model_dir and model_dir.exists()),
            tokenizer_kwargs={"fix_mistral_regex": True},
        )
        embeddings = model.encode_document(
            ["passage: " + text for text in semantic_documents],
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(output_dir / "semantic_embeddings.npy", embeddings)
        (output_dir / "semantic_mapping.json").write_text(
            json.dumps(semantic_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if model_dir and not (model_dir / "config.json").exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(model_dir))
        semantic_available = True

    # SOURCE_DATE_EPOCH permite manifiestos idénticos entre reconstrucciones.
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    created = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
    artifact_files = [
        "vectorizer.joblib", "lexical_profiles.npz", "evidence_matrix.npz",
        "town_term_rates.npz", "bm25.npz", "bm25.idf.npy", "metadata.json",
    ] + (["semantic_embeddings.npy", "semantic_mapping.json"] if semantic_available else [])
    manifest = {
        "created_at_utc": created.isoformat(),
        "dataset": str(data_path),
        "dataset_sha256": _sha256(data_path),
        "rows": int(len(df)),
        "rows_in_csv": total_rows,
        "holdout_rows_excluded": total_rows - int(len(df)),
        "destinations": len(towns),
        "regions": int(df["Region"].nunique()),
        "types": sorted(df["Type"].unique().tolist()),
        "vocabulary_size": int(len(feature_names)),
        "profile_count": int(lexical_profiles.shape[0]),
        "evidence_count": int(len(evidence)),
        "semantic_available": semantic_available,
        "semantic_model": model_name if semantic_available else None,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "runtime_versions": runtime_versions(),
        "seed": seed,
        "data_policy": "Solo se leyó la partición RestMex 2025 train indicada en dataset.",
        # .npz y .joblib incluyen metadatos de compresión; la reproducibilidad se verifica por
        # rankings (ver evaluation_report.json → determinism) además de estos hashes.
        "artifacts_sha256": {name: _sha256(output_dir / name) for name in artifact_files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest

