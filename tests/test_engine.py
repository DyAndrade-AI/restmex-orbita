from pathlib import Path

import pytest

from restmex_recommender import Recommender

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    if not (ROOT / "artifacts/vectorizer.joblib").exists():
        pytest.skip("Índice no construido")
    return Recommender(ROOT / "artifacts", enable_semantic=False)


def test_contract_returns_disjoint_rankings(engine):
    result = engine.recommend("Quiero playa, agua clara y snorkel")
    assert len(result["top"]) == 5
    assert len(result["bottom"]) == 5
    assert not {x["town"] for x in result["top"]} & {x["town"] for x in result["bottom"]}
    assert [x["rank"] for x in result["top"]] == [1, 2, 3, 4, 5]


def test_results_are_deterministic(engine):
    query = "Arquitectura colonial, historia y comida tradicional"
    first = [x["town"] for x in engine.recommend(query)["top"]]
    second = [x["town"] for x in engine.recommend(query)["top"]]
    assert first == second


def test_short_query_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.recommend("playa")


def test_future_constellation_has_distinct_roles_and_anchor(engine):
    query = "Quiero bosque, montaña, cabañas y comida tradicional"
    future = engine.recommend_futures(query, risk=0.6)
    classic = engine.recommend(query)

    assert len(future["destinations"]) == 5
    assert len({item["town"] for item in future["destinations"]}) == 5
    assert [item["role"] for item in future["destinations"]] == [
        "Ancla",
        "Superviviente",
        "Portal",
        "Especialista",
        "Comodín",
    ]
    assert future["destinations"][0]["town"] == classic["top"][0]["town"]
    assert len(future["facets"]) >= 2


def test_future_constellation_is_deterministic(engine):
    query = "Quiero cenotes, naturaleza, comida y tranquilidad"
    first = engine.recommend_futures(query, risk=0.8)
    second = engine.recommend_futures(query, risk=0.8)
    assert [item["town"] for item in first["destinations"]] == [
        item["town"] for item in second["destinations"]
    ]


def test_future_constellation_rejects_invalid_risk(engine):
    with pytest.raises(ValueError):
        engine.recommend_futures("Quiero playa y comida", risk=1.5)


def test_future_facets_remove_query_boilerplate(engine):
    result = engine.recommend_futures("Quiero bosque, aventura y gastronomía")
    labels = {item["label"] for item in result["facets"]}
    assert "quiero" not in labels


def test_multiverse_simulates_and_collapses_futures(engine):
    query = "Quiero bosque, cabañas, gastronomía y aventura, sin playa"
    first = engine.explore_multiverse(query, risk=0.6, futures=5_000)

    assert first["total_futures"] == 5_000
    assert first["surviving_futures"] == 5_000
    assert len(first["destinations"]) == 5
    assert first["question"] is not None
    assert sum(item["share"] for item in first["destinations"]) <= 100.01

    branch = first["question"]["branches"][0]
    collapsed = engine.explore_multiverse(
        query,
        risk=0.6,
        futures=5_000,
        choices=[{"keep": branch["keep"], "sacrifice": branch["sacrifice"]}],
    )
    assert 0 < collapsed["surviving_futures"] < first["surviving_futures"]
    assert collapsed["collapsed_futures"] > 0
    assert collapsed["choices"] == [
        {"keep": branch["keep"], "sacrifice": branch["sacrifice"]}
    ]


def test_multiverse_is_deterministic(engine):
    query = "Quiero historia, comida tradicional y arquitectura"
    first = engine.explore_multiverse(query, futures=5_000)
    second = engine.explore_multiverse(query, futures=5_000)
    assert first["seed"] == second["seed"]
    assert first["destinations"] == second["destinations"]

