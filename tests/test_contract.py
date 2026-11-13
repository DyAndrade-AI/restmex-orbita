"""Casos adversariales, empates, integridad de artefactos y paridad con Docker."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from restmex_recommender import Recommender
from restmex_recommender.build import build_index
from restmex_recommender.evaluation import (
    _mask_terms,
    full_ranking,
    mask_entities,
    ranking_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    if not (ROOT / "artifacts/vectorizer.joblib").exists():
        pytest.skip("Índice no construido")
    return Recommender(ROOT / "artifacts", enable_semantic=False)


def _towns(items):
    return [x["town"] for x in items]


def test_boundary_ties_keep_top_and_bottom_disjoint_and_consistent(engine, monkeypatch):
    original = engine._score_query

    def tied(query, *args, **kwargs):
        scored = original(query, *args, **kwargs)
        scores = np.zeros(len(engine.towns), dtype=np.float32)
        scores[:3] = 1.0  # tres ganadores y 37 empatados: el corte del top cae dentro del bloque
        return {**scored, "scores": scores}

    monkeypatch.setattr(engine, "_score_query", tied)
    result = engine.recommend("Quiero playa y snorkel")
    top, bottom = _towns(result["top"]), _towns(result["bottom"])
    assert not set(top) & set(bottom)
    ranking = full_ranking(engine, "Quiero playa y snorkel")
    assert top == ranking[:5]
    assert bottom == ranking[::-1][:5]  # el bottom mostrado es el que evalúan las métricas


def test_out_of_vocabulary_query_is_rejected_in_lexical_mode(engine):
    with pytest.raises(ValueError, match="arbitrario"):
        engine.recommend("xyzzy qwerty plugh")


def test_named_exclusion_sinks_the_town(engine):
    result = engine.recommend("Cenotes y comida yucateca, sin Tulum")
    assert "Tulum" not in _towns(result["top"])


def test_markup_and_emoji_queries_are_served(engine):
    for query in ("<script>alert(1)</script> playa con snorkel", "🏖️ 🌊 playa mar"):
        result = engine.recommend(query)
        assert len(result["top"]) == 5 and len(result["bottom"]) == 5


def test_bottom_never_uses_positive_reviews_as_justification(engine):
    result = engine.recommend("Quiero bosque, cabañas y senderismo")
    assert all(item["evidence"] is None for item in result["bottom"])
    assert all(item["relation"] == "menor afinidad temática" for item in result["bottom"])


def test_quality_is_reported_but_not_used_for_ranking(engine):
    query = "Arquitectura colonial e iglesias"
    before = full_ranking(engine, query)
    saved = engine.quality.copy()
    engine.quality[:] = np.linspace(1, 5, len(saved))
    try:
        assert full_ranking(engine, query) == before
    finally:
        engine.quality[:] = saved
    item = engine.recommend(query)["top"][0]
    assert 1.0 <= item["quality_bayes"] <= 5.0


def test_corpus_check_reports_mention_rates(engine):
    item = engine.recommend("Quiero cenotes para nadar")["top"][0]
    assert item["corpus_check"], "falta town_term_rates.npz"
    assert all(0 <= row["rate"] <= 100 for row in item["corpus_check"])


def test_ranking_metrics_names_are_correct():
    ranking = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    one = ranking_metrics(ranking, {"C"})
    assert one["precision_at_5"] == pytest.approx(0.2)
    assert one["capped_recall_at_5"] == 1.0
    assert one["mrr"] == pytest.approx(1 / 3)
    tail = ranking_metrics(ranking, {"J"}, excluded={"A"})
    assert tail["relevant_in_bottom"] == 1.0 and tail["exclusion_violation"] == 1.0


def test_holdout_masking_removes_town_and_region_names(engine):
    masked = mask_entities("Hermoso Tulum en Quintana Roo, cenote increíble", _mask_terms(engine))
    assert "tulum" not in masked and "cenote" in masked


def test_stale_semantic_artifacts_are_ignored_after_lexical_rebuild(tmp_path, monkeypatch):
    monkeypatch.delenv("ORBITA_SEMANTIC", raising=False)  # que decida el manifest, no el entorno
    csv = ROOT / "data/Rest-Mex_2025_train_clean.csv"
    if not csv.exists():
        pytest.skip("CSV no disponible")
    sample = pd.read_csv(csv, nrows=4000)
    small = tmp_path / "small.csv"
    sample.to_csv(small, index=False)
    out = tmp_path / "artifacts"
    build_index(small, out, max_features=5_000)
    # Simula artefactos de una construcción semántica anterior en el mismo directorio.
    np.save(out / "semantic_embeddings.npy", np.zeros((4, 8), dtype=np.float32))
    (out / "semantic_mapping.json").write_text(json.dumps([]), encoding="utf-8")
    rebuilt = Recommender(out, tmp_path / "no-model")
    assert not rebuilt.semantic_enabled
    assert "ignorados" in rebuilt.semantic_status
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["semantic_available"] is False
    assert "town_term_rates.npz" in manifest["artifacts_sha256"]


def _load_api():
    spec = importlib.util.spec_from_file_location("serve_api", ROOT / "scripts/serve_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labels_require_a_token_issued_for_that_ranking(engine):
    api = _load_api()
    result = engine.recommend("Quiero playa y snorkel")
    top, bottom = _towns(result["top"]), _towns(result["bottom"])
    payload = {
        "query": "Quiero playa y snorkel",
        "mode": result["mode"],
        "top": top,
        "bottom": bottom,
        "judgments": {t: 1 for t in top} | {t: 0 for t in bottom},
        "token": api.ranking_token("Quiero playa y snorkel", result["mode"], top, bottom),
    }
    record = api.validate_labels(payload, set(engine.towns))
    assert record["judgments"][top[0]] == 1
    forged = {**payload, "top": bottom, "bottom": top}
    with pytest.raises(ValueError, match="no coincide"):
        api.validate_labels(forged, set(engine.towns))


def test_docker_image_uses_the_same_source_and_artifacts():
    dockerfile = (ROOT / "docker/api.Dockerfile").read_text(encoding="utf-8")
    assert not (ROOT / "docker/runtime").exists(), "código duplicado divergente"
    assert "COPY src/restmex_recommender/" in dockerfile
    for artifact in ("vectorizer.joblib", "lexical_profiles.npz", "evidence_matrix.npz",
                     "town_term_rates.npz", "bm25.npz", "bm25.idf.npy", "metadata.json",
                     "manifest.json"):
        assert artifact in dockerfile, artifact


def test_manifest_matches_dataset_and_artifacts():
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text(encoding="utf-8"))
    csv = ROOT / "data/Rest-Mex_2025_train_clean.csv"
    if not csv.exists():
        pytest.skip("CSV no disponible")
    from restmex_recommender.build import _sha256

    assert _sha256(csv) == manifest["dataset_sha256"]
    for name, digest in manifest["artifacts_sha256"].items():
        assert _sha256(ROOT / "artifacts" / name) == digest, name


def test_query_with_only_exclusions_is_rejected_instead_of_inverted(engine):
    with pytest.raises(ValueError, match="exclusiones"):
        engine.recommend("sin playa, sin bosque, sin comida")


def test_spelling_correction_is_reported_and_leaves_known_words(engine):
    result = engine.recommend("plallas con snorkell y agua cristalina")
    corrections = {c["from"]: c["to"] for c in result["confidence"]["spelling_corrections"]}
    assert corrections == {"plallas": "playas", "snorkell": "snorkel"}
    assert engine.correct_spelling("bosque y cabañas")[1] == []


def test_rankings_move_coherently_when_the_query_changes(engine):
    base = full_ranking(engine, "Arquitectura colonial e iglesias")
    with_beach = full_ranking(engine, "Arquitectura colonial e iglesias junto a la playa")
    without_beach = full_ranking(engine, "Arquitectura colonial e iglesias, sin playa")
    for town in ("Tulum", "Isla_Mujeres", "Sayulita"):
        assert with_beach.index(town) < base.index(town) <= without_beach.index(town)
