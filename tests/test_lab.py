"""Laboratorio de evaluación humana: acuerdo, particiones, pools ciegos y métricas."""

import json
from pathlib import Path

import numpy as np
import pytest

from restmex_recommender import Recommender
from restmex_recommender.lab import (
    Lab,
    cohen_kappa_quadratic,
    human_metrics,
    krippendorff_alpha,
    load_bank,
    split_for,
    tuning_queries,
)

ROOT = Path(__file__).resolve().parents[1]

# Ejemplo canónico de Krippendorff (2011), 4 codificadores × 12 unidades, None = faltante.
KRIPPENDORFF_UNITS = [
    [1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
    [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1],
    [None, 3, None, None],
]


@pytest.mark.parametrize("metric,expected", [
    ("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849),
])
def test_krippendorff_alpha_matches_published_example(metric, expected):
    assert krippendorff_alpha(KRIPPENDORFF_UNITS, metric) == pytest.approx(expected, abs=1e-3)


def test_quadratic_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(3)
    a = rng.integers(0, 4, 60).tolist()
    b = [min(3, max(0, x + int(rng.integers(-1, 2)))) for x in a]
    expected = cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3])
    assert cohen_kappa_quadratic(a, b) == pytest.approx(expected, abs=1e-9)


def test_splits_are_stable_and_test_is_never_used_for_tuning():
    bank = load_bank(ROOT / "eval/query_bank.json")
    assert all(q["split"] == split_for(q["id"]) for q in bank)
    assert {q["split"] for q in bank} == {"dev", "calibration", "test"}
    assert all(q["split"] != "test" for q in tuning_queries(bank))


def test_human_metrics_treat_unjudged_as_unknown():
    ranking = [f"T{i}" for i in range(40)]
    judged = {"T0": {"relevance": 3, "violates": False}, "T1": {"relevance": 0, "violates": True},
              "T39": {"relevance": 2, "violates": False}, "T20": {"relevance": 3, "violates": False}}
    m = human_metrics(ranking, judged)
    assert m["judged_at_5"] == pytest.approx(0.4)
    assert m["precision_at_5"] == pytest.approx(0.5)  # 1 de 2 juzgados con rel ≥ 2
    assert m["top5_constraint_errors"] == pytest.approx(0.5)
    assert m["bottom5_misplaced"] == 1.0
    assert m["false_negatives_middle"] == 1.0
    ideal = 3 + 3 / np.log2(3) + 2 / np.log2(4) + 0
    assert m["ndcg_at_5"] == pytest.approx(3 / ideal)


@pytest.fixture()
def lab(tmp_path):
    if not (ROOT / "artifacts/vectorizer.joblib").exists():
        pytest.skip("Índice no construido")
    (tmp_path / "query_bank.json").write_text(
        (ROOT / "eval/query_bank.json").read_text(encoding="utf-8"), encoding="utf-8")
    return Lab(Recommender(ROOT / "artifacts", enable_semantic=False), tmp_path)


def test_pool_contains_both_ends_of_each_system_and_task_is_blind(lab):
    qid = "colonial-cultura-sin-playa"
    pool = lab.pool(qid)
    ranking = lab.engine.recommend(lab.by_id[qid]["query"])
    assert {x["town"] for x in ranking["top"] + ranking["bottom"]} <= set(pool["towns"])
    task = lab.task(qid, "evaluador-a")
    text = json.dumps(task)
    assert "score" not in text and "rank" not in text and "system" not in text
    assert sorted(d["town"] for d in task["destinations"]) == pool["towns"]
    other = lab.task(qid, "evaluador-b")
    assert [d["town"] for d in task["destinations"]] != [d["town"] for d in other["destinations"]]


def test_save_requires_complete_valid_judgments_and_reports_agreement(lab):
    qid = "colonial-cultura-sin-playa"
    towns = lab.pool(qid)["towns"]
    with pytest.raises(ValueError, match="todos"):
        lab.save({"evaluator": "a", "query_id": qid,
                  "judgments": [{"town": towns[0], "relevance": 3}]})
    with pytest.raises(ValueError, match="0, 1, 2 o 3"):
        lab.save({"evaluator": "a", "query_id": qid,
                  "judgments": [{"town": t, "relevance": 7} for t in towns]})
    for evaluator, shift in (("a", 0), ("b", 1)):
        lab.save({"evaluator": evaluator, "query_id": qid, "seconds": 40, "judgments": [
            {"town": t, "relevance": min(3, (i + shift) % 4), "constraint": "cumple"}
            for i, t in enumerate(towns)]})
    report = lab.report()
    assert report["judgments"] == 2 * len(towns)
    assert report["agreement"]["items_with_2plus_raters"] == len(towns)
    assert report["agreement"]["krippendorff_alpha_ordinal"] is not None
    assert "lexico" in report["by_split_and_system"][lab.by_id[qid]["split"]]
    assert "relevance" in lab.export("csv").splitlines()[0]


def test_test_split_queue_is_random_and_skips_done_queries(lab):
    queue = lab.queue("a", "test", limit=50)
    assert queue and all(item["reason"] == "aleatorio" for item in queue)
    assert all(lab.by_id[item["query_id"]]["split"] == "test" for item in queue)
    dev = lab.queue("a", "dev", limit=50)
    assert len({item["query_id"] for item in dev}) == len(dev)


def test_contradictory_query_is_flagged():
    engine = Recommender(ROOT / "artifacts", enable_semantic=False)
    result = engine.recommend("Quiero playa pero sin playa ni mar")
    assert result["contradictions"] and result["confidence"]["level"] == "baja"
