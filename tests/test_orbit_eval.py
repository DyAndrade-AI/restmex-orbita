"""Átomos de gusto, usuarios simulados de Strange Loop y sensibilidad de los roles."""

from pathlib import Path

import pytest

from restmex_recommender import Recommender
from restmex_recommender.futures import _extract_facets
from restmex_recommender.orbit_eval import role_sensitivity, simulate_users
from restmex_recommender.text import parse_query

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    if not (ROOT / "artifacts/vectorizer.joblib").exists():
        pytest.skip("Índice no construido")
    return Recommender(ROOT / "artifacts", enable_semantic=False)


def labels(engine, query):
    return [f["label"] for f in _extract_facets(engine, parse_query(query).positive)]


@pytest.mark.parametrize("query, expected", [
    ("Viaje con niños pequeños, actividades para la familia", ["familia"]),
    ("Aventura extrema: tirolesa, rappel y cuatrimoto", ["aventura"]),
    ("Algo romántico para nuestro aniversario", ["romántico"]),
    ("Vida nocturna, bares y fiesta", ["vida nocturna"]),
    ("Café de altura y artesanías", ["café", "artesanía"]),
])
def test_synonyms_form_one_atom(engine, query, expected):
    assert labels(engine, query) == expected


def test_concrete_atoms_unchanged(engine):
    # El ejemplo del video: los sustantivos concretos siguen saliendo del IDF.
    assert labels(engine, "Busco arquitectura colonial, museos, mezcal y naturaleza, sin playa") == [
        "arquitectura / colonial", "museos", "mezcal", "naturaleza"]


def test_simulated_users_curves(engine):
    row = simulate_users(engine, "Busco cenotes para nadar y naturaleza", users=20,
                         futures=4_000, max_questions=2)
    info, rand = row["curves"]["informativa"], row["curves"]["azar"]
    assert info["hit_at_1"][0] == rand["hit_at_1"][0]  # sin preguntas, misma respuesta
    assert all(0 <= v <= 1 for v in info["hit_at_1"] + rand["hit_at_5"])
    assert info["mean_questions_asked"] <= 2


def test_single_facet_has_no_questions(engine):
    assert simulate_users(engine, "Vida nocturna, bares y fiesta", users=5, futures=2_000) is None


def test_zero_perturbation_keeps_every_role(engine):
    out = role_sensitivity(engine, ["Quiero bosque, cabañas, gastronomía y aventura"], spread=0.0, draws=3)
    assert set(out["role_unchanged_pct"].values()) == {100.0}
    assert out["portfolio_jaccard_mean"] == 1.0
