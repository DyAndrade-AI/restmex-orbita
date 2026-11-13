"""Validación matemática de Órbita y Strange Loop."""

from pathlib import Path

import numpy as np
import pytest

from restmex_recommender import Recommender
from restmex_recommender.futures import _counterfactual, _entropy

ROOT = Path(__file__).resolve().parents[1]
QUERY = "Quiero arquitectura colonial, gastronomía tradicional y actividades culturales, pero sin playa"


@pytest.fixture(scope="module")
def engine():
    if not (ROOT / "artifacts/vectorizer.joblib").exists():
        pytest.skip("Índice no construido")
    return Recommender(ROOT / "artifacts", enable_semantic=False)


def test_entropy_extremes():
    assert _entropy(np.zeros(100, dtype=int), 40) == pytest.approx(0.0)
    assert _entropy(np.arange(40).repeat(25), 40) == pytest.approx(1.0)
    two = _entropy(np.array([0, 1] * 50), 40)
    assert two == pytest.approx(np.log(2) / np.log(40))
    assert _entropy(np.array([], dtype=int), 40) == 0.0


def test_counterfactual_shift_solves_the_tie_equation():
    # Facetas × destinos. El destino 1 gana solo en la faceta 0.
    matrix = np.array([[0.2, 0.9], [0.8, 0.1], [0.7, 0.2]])
    cf = _counterfactual(matrix, [{"label": "a"}, {"label": "b"}, {"label": "c"}], 0, 1)
    diff = matrix[:, 1] - matrix[:, 0]
    s = -diff.mean() / (diff.max() - diff.mean())
    assert cf["facet"] == "a" and cf["priority_shift"] == int(np.ceil(s * 100))
    weaker = _counterfactual(np.array([[0.9, 0.1], [0.8, 0.2]]), [{"label": "a"}, {"label": "b"}], 0, 1)
    assert weaker["priority_shift"] is None


def test_information_gain_and_collapse_are_consistent(engine):
    query = "Quiero bosque, cabañas, gastronomía y aventura, sin playa"
    first = engine.explore_multiverse(query, futures=20_000)
    question = first["question"]
    assert question is not None
    branches = question["branches"]
    assert sum(b["survivors"] for b in branches) == first["surviving_futures"]
    entropies, sizes = [], []
    for branch in branches:
        collapsed = engine.explore_multiverse(
            query, futures=20_000, choices=[{"keep": branch["keep"], "sacrifice": branch["sacrifice"]}]
        )
        # El colapso conserva exactamente los futuros que la pregunta anunció para esa rama.
        assert collapsed["surviving_futures"] == branch["survivors"]
        entropies.append(collapsed["decision_entropy"] / 100)
        sizes.append(branch["survivors"])
    p = np.array(sizes) / sum(sizes)
    expected_gain = first["decision_entropy"] / 100 - float(p @ np.array(entropies))
    assert question["information_gain"] / 100 == pytest.approx(expected_gain, abs=2e-4)
    assert question["information_gain"] >= 0


def test_monte_carlo_uncertainty_is_reported_correctly(engine):
    result = engine.explore_multiverse(QUERY, futures=20_000)
    n = result["surviving_futures"]
    for d in result["destinations"]:
        p = d["share"] / 100
        assert d["share_se"] == pytest.approx(100 * np.sqrt(p * (1 - p) / n), abs=2e-3)
        assert d["share_ci95"][0] <= d["share"] <= d["share_ci95"][1]
    assert result["monte_carlo"]["worst_case_se_pp"] == pytest.approx(50 / np.sqrt(n), abs=1e-3)
    assert result["monte_carlo"]["convergence"][-1]["max_abs_share_diff_vs_full_pp"] == 0


def test_constraint_blocked_towns_never_win(engine):
    result = engine.explore_multiverse(QUERY, futures=10_000)
    blocked = {b["town"] for b in result["bottom_in_multiverse"] if b["blocked_by_constraint"]}
    assert blocked, "se esperaban pueblos de playa bloqueados"
    assert not blocked & {d["town"] for d in result["destinations"]}
    assert all(b["stays_bottom5_pct"] is None for b in result["bottom_in_multiverse"]
               if b["blocked_by_constraint"])


def test_orbita_anchor_bottom_view_and_stability(engine):
    classic = engine.recommend(QUERY)
    orbit = engine.recommend_futures(QUERY)
    assert orbit["destinations"][0]["town"] == classic["top"][0]["town"]
    assert [b["town"] for b in orbit["bottom_perspective"]] == [x["town"] for x in classic["bottom"]]
    assert all(0 <= d["stability"] <= 100 for d in orbit["destinations"])
    assert "Superviviente" in orbit["method"]["definitions"]
    # Los roles no incluyen destinos con evidencia de incumplir la restricción (salvo el Ancla).
    violators = {x["town"] for x in classic["top"] + classic["bottom"] if x["violated"]}
    assert not violators & {d["town"] for d in orbit["destinations"][1:]}


def test_multiverse_rejects_invalid_collapse(engine):
    with pytest.raises(ValueError):
        engine.explore_multiverse(QUERY, futures=5_000, choices=[{"keep": "x", "sacrifice": "y"}])
