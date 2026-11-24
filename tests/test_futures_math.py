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


LABELS = [{"label": "a"}, {"label": "b"}, {"label": "c"}]


def _gap(matrix, relevance, anchor, dest, facet, shift):
    k = matrix.shape[0]
    w = np.full(k, (1 - shift) / k)
    w[facet] += shift
    return 0.16 * (relevance[dest] - relevance[anchor]) + 0.84 * w @ (matrix[:, dest] - matrix[:, anchor])


def test_counterfactual_shift_is_sufficient_and_minimal():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(500):
        matrix, relevance = rng.random((3, 6)), rng.random(6)
        anchor = int(np.argmax(relevance))
        for dest in range(6):
            cf = _counterfactual(matrix, relevance, LABELS, anchor, dest)
            f = [x["label"] for x in LABELS].index(cf["facet"])
            s = cf["priority_shift"]
            if s is None:
                # Imposible para CUALQUIER ponderación: ni siquiera el mejor vértice del símplex.
                best = max(_gap(matrix, relevance, anchor, dest, j, 1.0) for j in range(3))
                assert best < 0
            elif s == 0:
                assert _gap(matrix, relevance, anchor, dest, f, 0.0) >= 0
            else:
                checked += 1
                assert _gap(matrix, relevance, anchor, dest, f, s / 100) >= 0
                # Minimalidad sobre toda dirección v (no solo la faceta elegida).
                assert all(_gap(matrix, relevance, anchor, dest, j, (s - 1) / 100) < 0 for j in range(3))
            assert cf == _counterfactual(matrix, relevance, LABELS, anchor, dest)  # determinista
    assert checked > 50


def test_counterfactual_against_itself_is_zero():
    matrix = np.array([[0.3, 0.1], [0.2, 0.9]])
    assert _counterfactual(matrix, np.array([1.0, 0.0]), LABELS[:2], 0, 0)["priority_shift"] == 0


def test_roles_respect_hard_exclusions_and_are_distinct(engine):
    from restmex_recommender.futures import constellation_inputs, select_roles
    for query in ["Quiero arquitectura colonial y gastronomía, sin playa",
                  "Quiero bosque, cabañas, gastronomía y aventura, sin playa",
                  "Pueblos con cenotes, sin Tulum"]:
        for risk in (0.0, 0.55, 1.0):
            x = constellation_inputs(engine, query, risk)
            # Forzar la peor situación: nadie cumple el umbral blando.
            x["eligible"][:] = False
            x["eligible"][x["anchor"]] = True
            sel = select_roles(x, risk)
            assert len(set(sel)) == len(sel) == 5
            assert not x["unsafe"][sel].any()
        tulum = [i for i, d in enumerate(engine.destinations) if d["town"] == "Tulum"]
        if "Tulum" in query and tulum:
            assert x["unsafe"][tulum[0]]


def test_too_few_safe_destinations_shrinks_instead_of_breaking(engine):
    from restmex_recommender.futures import constellation_inputs, select_roles
    x = constellation_inputs(engine, "Quiero bosque y gastronomía, sin playa", 0.55)
    x["unsafe"][:] = True
    keep = np.argsort(-x["relevance"])[:3]
    x["unsafe"][keep] = False
    x["anchor"] = int(keep[0])
    sel = select_roles(x, 0.55)
    assert sorted(sel) == sorted(keep.tolist())


def test_constellation_keeps_5_plus_5_and_honest_copy(engine):
    out = engine.recommend_futures(QUERY)
    ranking = engine.recommend(QUERY)
    assert out["destinations"][0]["town"] == ranking["top"][0]["town"]
    assert len({d["town"] for d in out["destinations"]}) == 5
    assert len(out["bottom_perspective"]) == 5
    for row in out["bottom_perspective"]:
        assert "supera al Ancla si solo importa" not in row["verdict"]


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
