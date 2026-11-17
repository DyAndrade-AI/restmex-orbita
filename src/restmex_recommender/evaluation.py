"""Evaluación automática del contrato 5 + 5.

Tres fuentes de verdad, de menor a mayor independencia respecto al autor del sistema:
1. ``gold``: seed heurístico curado (eval/gold_queries.json), pendiente de adjudicación humana;
   sirve como regresión, no como evidencia no sesgada (su autor vio resultados previos).
2. ``holdout``: tarea autosupervisada. Se reservan reseñas del train, se reconstruye el índice SIN
   ellas y cada reseña (con nombres de pueblo y región enmascarados) se usa como consulta; el
   destino correcto es el pueblo que la escribió. No depende de consultas escritas a mano.
3. ``human``: etiquetas ciegas capturadas en la interfaz de evaluación (eval/human_labels.jsonl).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .engine import Recommender, _rank_order
from .text import canonical_text

RUBRIC = {
    "precision_at_5": 0.70,
    "hit_at_1": 0.80,
    "contract": 1.0,
    "determinism": 1.0,
    "latency_p95_s": 3.0,
}


# ---------------------------------------------------------------- métricas por consulta
def ranking_metrics(
    ranking: Sequence[str],
    relevant: Iterable[str],
    excluded: Iterable[str] = (),
    k: int = 5,
) -> dict:
    """Métricas binarias sobre un ranking completo de destinos."""
    relevant, excluded = set(relevant), set(excluded)
    top, bottom = list(ranking[:k]), list(ranking[-k:])
    hits = [1.0 if town in relevant else 0.0 for town in top]
    first = next((i for i, t in enumerate(ranking) if t in relevant), None)
    ideal = min(k, len(relevant))
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return {
        "precision_at_5": sum(hits) / k,
        "recall_at_5": sum(hits) / len(relevant) if relevant else 0.0,
        # Recall acotado: hits / min(5, |R|). Con 1 relevante, acertarlo vale 1.0.
        "capped_recall_at_5": sum(hits) / ideal if ideal else 0.0,
        "relevant_count": float(len(relevant)),
        "hit_at_1": hits[0] if hits else 0.0,
        "hit_at_5": 1.0 if any(hits) else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg_at_5": dcg / idcg if idcg else 0.0,
        "relevant_in_bottom": float(len(relevant.intersection(bottom))),
        "exclusion_violation": 1.0 if excluded.intersection(top) else 0.0,
    }


def _mean(rows: List[dict], key: str) -> float:
    return round(float(np.mean([r[key] for r in rows])), 4) if rows else 0.0


def aggregate(rows: List[dict]) -> dict:
    keys = [
        "precision_at_5", "recall_at_5", "capped_recall_at_5", "hit_at_1", "hit_at_5", "mrr",
        "ndcg_at_5",
    ]
    out = {key: _mean(rows, key) for key in keys}
    # Precision@5 solo es comparable con el umbral cuando hay al menos 5 relevantes posibles.
    wide = [r for r in rows if r.get("relevant_count", 0) >= 5]
    out["precision_at_5_wide"] = _mean(wide, "precision_at_5")
    out["wide_queries"] = len(wide)
    out["bottom_error_rate"] = (
        round(float(np.mean([r["relevant_in_bottom"] > 0 for r in rows])), 4) if rows else 0.0
    )
    with_exclusion = [r for r in rows if r.get("has_exclusion")]
    out["exclusion_violation_rate"] = _mean(with_exclusion, "exclusion_violation")
    out["queries"] = len(rows)
    return out


def full_ranking(engine: Recommender, query: str, weights: Optional[dict] = None) -> List[str]:
    scores = engine._score_query(query, weights)["scores"]
    return [engine.towns[i] for i in _rank_order(scores)]


def _percentile(values: List[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 4) if values else 0.0


# ---------------------------------------------------------------- suites
def load_gold(path: Path) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["queries"]


def gold_suite(engine: Recommender, gold: List[dict]) -> dict:
    unknown = {t for q in gold for t in q["relevant"] + q.get("excluded", [])} - set(engine.towns)
    if unknown:
        raise ValueError(f"El gold set menciona pueblos inexistentes: {sorted(unknown)}")
    rows, latencies, contract_ok = [], [], []
    for item in gold:
        start = time.perf_counter()
        result = engine.recommend(item["query"])
        latencies.append(time.perf_counter() - start)
        top = [x["town"] for x in result["top"]]
        bottom = [x["town"] for x in result["bottom"]]
        contract_ok.append(len(top) == 5 and len(bottom) == 5 and not set(top) & set(bottom))
        ranking = full_ranking(engine, item["query"])
        metrics = ranking_metrics(ranking, item["relevant"], item.get("excluded", []))
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "query": item["query"],
                "top": top,
                "bottom": bottom,
                "relevant": item["relevant"],
                "has_exclusion": bool(item.get("excluded")),
                "confidence": result["confidence"]["level"],
                **{k: round(v, 4) for k, v in metrics.items()},
            }
        )
    by_category: Dict[str, List[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    return {
        "aggregate": aggregate(rows),
        "by_category": {c: aggregate(r) for c, r in sorted(by_category.items())},
        "contract_pass_rate": round(float(np.mean(contract_ok)), 4),
        "latency_s": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": round(max(latencies), 4),
        },
        "queries": rows,
    }


def baseline_suite(engine: Recommender, gold: List[dict], seed: int = 2025) -> dict:
    """Referencias mínimas: azar y popularidad (más reseñas primero)."""
    rng = np.random.default_rng(seed)
    by_volume = sorted(
        engine.towns, key=lambda t: -engine.destinations[engine.town_to_idx[t]]["review_count"]
    )
    random_rows, popularity_rows = [], []
    for item in gold:
        rel, exc = item["relevant"], item.get("excluded", [])
        for _ in range(200):
            perm = list(rng.permutation(engine.towns))
            random_rows.append({**ranking_metrics(perm, rel, exc), "has_exclusion": bool(exc)})
        popularity_rows.append({**ranking_metrics(by_volume, rel, exc), "has_exclusion": bool(exc)})
    return {"random": aggregate(random_rows), "popularity": aggregate(popularity_rows)}


def ablation_suite(engine: Recommender, gold: List[dict]) -> dict:
    """Contribución de cada señal: mismas consultas, pesos distintos."""
    from .engine import (
        HYBRID_WEIGHTS, LEXICAL_WEIGHTS, PREVIOUS_HYBRID_WEIGHTS, PREVIOUS_LEXICAL_WEIGHTS,
    )

    configs = {
        "solo_perfil_lexico": {"lexical": 1.0, "evidence": 0.0, "semantic": 0.0},
        "solo_evidencia": {"lexical": 0.0, "evidence": 1.0, "semantic": 0.0},
        "solo_cobertura": {"coverage": 1.0},
        "lexico_anterior": dict(PREVIOUS_LEXICAL_WEIGHTS),
        "lexico_cobertura": dict(LEXICAL_WEIGHTS),
        "bm25": {"bm25": 1.0},
    }
    if engine.semantic_enabled:
        configs["solo_semantico"] = {"lexical": 0.0, "evidence": 0.0, "semantic": 1.0}
        configs["hibrido"] = dict(HYBRID_WEIGHTS)
        configs["hibrido_anterior"] = dict(PREVIOUS_HYBRID_WEIGHTS)
    out = {}
    # Pesos explícitos por llamada: el motor compartido no se modifica.
    for name, weights in configs.items():
        rows = []
        for item in gold:
            try:
                ranking = full_ranking(engine, item["query"], weights)
            except ValueError:
                ranking = list(engine.towns)  # sin señal: equivale a orden arbitrario
            rows.append(
                {
                    **ranking_metrics(ranking, item["relevant"], item.get("excluded", [])),
                    "has_exclusion": bool(item.get("excluded")),
                }
            )
        out[name] = aggregate(rows)
    return out


def _variants(query: str) -> List[str]:
    clauses = [c.strip() for c in re.split(r",| y ", query) if c.strip()]
    return [
        canonical_text(query).upper(),  # sin acentos y en mayúsculas
        "por favor, " + query + ", gracias",  # relleno
        ", ".join(reversed(clauses)),  # orden de cláusulas invertido
    ]


def robustness_suite(engine: Recommender, gold: List[dict]) -> dict:
    """Jaccard del top 5 entre la consulta original y variantes superficiales."""
    scores = []
    for item in gold:
        base = {x["town"] for x in engine.recommend(item["query"])["top"]}
        for variant in _variants(item["query"]):
            try:
                other = {x["town"] for x in engine.recommend(variant)["top"]}
            except ValueError:
                other = set()
            scores.append(len(base & other) / len(base | other))
    return {
        "mean_top5_jaccard": round(float(np.mean(scores)), 4),
        "min_top5_jaccard": round(float(np.min(scores)), 4),
        "pairs": len(scores),
    }


ADVERSARIAL = [
    ("vacía", "", "reject"),
    ("una palabra", "playa", "reject"),
    ("solo stopwords", "de la que el en", "reject_or_low"),
    ("fuera de vocabulario", "xyzzy qwerty plugh", "reject_or_low"),
    ("solo números", "12345 67890", "reject_or_low"),
    ("emoji", "🏖️ 🌊 playa mar", "serve"),
    ("html", "<script>alert(1)</script> playa con snorkel", "serve"),
    ("muy larga", " ".join(["bosque montaña cabañas"] * 300), "serve"),
    ("todo negado", "sin playa, sin bosque, sin comida", "reject"),
    ("pueblo excluido", "cenotes y comida yucateca sin Valladolid", "serve"),
]


def adversarial_suite(engine: Recommender) -> dict:
    cases = []
    for name, query, expected in ADVERSARIAL:
        try:
            result = engine.recommend(query)
            top = [x["town"] for x in result["top"]]
            bottom = [x["town"] for x in result["bottom"]]
            valid = len(top) == 5 and len(bottom) == 5 and not set(top) & set(bottom)
            outcome = f"servida (confianza {result['confidence']['level']})"
            ok = valid and (
                expected == "serve"
                or (expected == "reject_or_low" and result["confidence"]["level"] == "baja")
            )
            if name == "pueblo excluido":
                ok = ok and "Valladolid" not in top
        except ValueError as exc:
            outcome, ok = f"rechazada: {exc}", expected != "serve"
        cases.append({"case": name, "expected": expected, "outcome": outcome, "pass": ok})
    return {"pass_rate": round(float(np.mean([c["pass"] for c in cases])), 4), "cases": cases}


def determinism_suite(engine: Recommender, gold: List[dict]) -> dict:
    first = [full_ranking(engine, q["query"]) for q in gold]
    second = [full_ranking(engine, q["query"]) for q in gold]
    fingerprint = hashlib.sha256(json.dumps(first).encode("utf-8")).hexdigest()
    return {
        "identical_rate": round(float(np.mean([a == b for a, b in zip(first, second)])), 4),
        "ranking_fingerprint": fingerprint,
    }


def volume_bias_suite(engine: Recommender, gold_rows: List[dict]) -> dict:
    """¿Los pueblos con más reseñas aparecen más en el top 5? Spearman exposición vs volumen."""
    from scipy.stats import spearmanr

    exposure = {t: 0 for t in engine.towns}
    for row in gold_rows:
        for town in row["top"]:
            exposure[town] += 1
    counts = [engine.destinations[engine.town_to_idx[t]]["review_count"] for t in engine.towns]
    rho, pvalue = spearmanr([exposure[t] for t in engine.towns], counts)
    never = sorted(t for t, v in exposure.items() if v == 0)
    return {
        "spearman_exposure_vs_reviews": round(float(rho), 4),
        "p_value": round(float(pvalue), 4),
        "max_exposure": max(exposure.items(), key=lambda kv: kv[1]),
        "towns_never_in_top5": never,
        "exposure": exposure,
    }


# ---------------------------------------------------------------- holdout sin fuga
def _mask_terms(engine: Recommender) -> List[str]:
    terms = set()
    for item in engine.destinations:
        for raw in (item["display_name"], item["display_region"]):
            spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)  # TodosSantos -> Todos Santos
            for text in (raw, spaced):
                for token in canonical_text(text).split():
                    if len(token) >= 3:
                        terms.add(token)
                terms.add(canonical_text(text).replace(" ", ""))
    terms.update({"tequis", "cuatrocienegas", "sancris", "valladolid"})
    return sorted(terms, key=len, reverse=True)


def mask_entities(text: str, terms: Sequence[str]) -> str:
    canonical = canonical_text(text)
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
    return re.sub(r"\s+", " ", re.sub(pattern, " ", canonical)).strip()


def holdout_split(df, per_town: int, seed: int) -> np.ndarray:
    """Máscara de filas reservadas: reseñas positivas de longitud media, estratificadas por pueblo."""
    eligible = (df["Polarity"] >= 4) & df["Review"].fillna("").str.len().between(80, 600)
    mask = np.zeros(len(df), dtype=bool)
    for _, group in df[eligible].groupby("Town", sort=True):
        chosen = group.sample(min(per_town, len(group)), random_state=seed).index
        mask[df.index.get_indexer(chosen)] = True
    return mask


def prepare_holdout(
    data_path: Path,
    cache_dir: Path,
    *,
    per_town: int = 15,
    seed: int = 7,
    with_semantic: bool = False,
    model_dir: Optional[Path] = None,
    progress: Callable[[str], None] = print,
):
    """Construye (o reutiliza) el índice sin las reseñas reservadas y devuelve las consultas."""
    import pandas as pd
    import sklearn

    from .build import _sha256, build_index

    if per_town < 1:
        raise ValueError("per_town debe ser al menos 1")
    df = pd.read_csv(data_path)
    mask = holdout_split(df, per_town, seed)
    cache_dir = Path(cache_dir)
    package = Path(__file__).resolve().parent
    # La caché se invalida si cambia el CSV, la máscara, el código del constructor o sklearn.
    signature = hashlib.sha256(
        json.dumps(
            {
                "dataset": _sha256(Path(data_path)),
                "mask": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                "builder": [_sha256(package / f) for f in ("build.py", "text.py", "bm25.py")],
                "sklearn": sklearn.__version__,
                "semantic": with_semantic,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    marker = cache_dir / "holdout_signature.txt"
    manifest_path = cache_dir / "manifest.json"
    reusable = (
        marker.exists()
        and marker.read_text(encoding="utf-8") == signature
        and manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")).get("holdout_rows_excluded")
        == int(mask.sum())
    )
    if not reusable:
        progress(f"Construyendo índice sin {int(mask.sum())} reseñas reservadas…")
        build_index(
            data_path, cache_dir, with_semantic=with_semantic, model_dir=model_dir,
            exclude_rows=mask,
        )
        marker.write_text(signature, encoding="utf-8")
    engine = Recommender(cache_dir, model_dir, enable_semantic=with_semantic)
    if with_semantic and not engine.semantic_enabled:
        raise RuntimeError(f"Holdout semántico pedido pero E5 no cargó: {engine.semantic_status}")
    terms = _mask_terms(engine)
    held = df[mask]
    queries = [
        (mask_entities(f"{r['Title']} {r['Review']}", terms), r["Town"]) for _, r in held.iterrows()
    ]
    return engine, held, queries


def holdout_suite(
    data_path: Path,
    cache_dir: Path,
    *,
    per_town: int = 15,
    seed: int = 7,
    with_semantic: bool = False,
    model_dir: Optional[Path] = None,
    progress: Callable[[str], None] = print,
) -> dict:
    engine, held, queries = prepare_holdout(
        data_path, cache_dir, per_town=per_town, seed=seed, with_semantic=with_semantic,
        model_dir=model_dir, progress=progress,
    )
    rows, by_town = [], {}
    for query, town in queries:
        try:
            ranking = full_ranking(engine, query)
        except ValueError:
            continue
        metrics = ranking_metrics(ranking, [town])
        rows.append({**metrics, "has_exclusion": False})
        by_town.setdefault(town, []).append(metrics["hit_at_5"])

    from scipy.stats import spearmanr

    towns = sorted(by_town)
    counts = [engine.destinations[engine.town_to_idx[t]]["review_count"] for t in towns]
    rho, _ = spearmanr([np.mean(by_town[t]) for t in towns], counts)
    by_volume = sorted(
        engine.towns, key=lambda t: -engine.destinations[engine.town_to_idx[t]]["review_count"]
    )
    popularity = aggregate(
        [{**ranking_metrics(by_volume, [r["Town"]]), "has_exclusion": False}
         for _, r in held.iterrows()]
    )
    n = len(engine.towns)
    return {
        "description": (
            "Cada reseña reservada (no vista al construir el índice) actúa como consulta con "
            "nombres de pueblo y región enmascarados; el destino correcto es su pueblo."
        ),
        "mode": "híbrido" if engine.semantic_enabled else "léxico",
        "held_out_reviews": len(held),
        "evaluated_queries": len(rows),
        "metrics": aggregate(rows),
        "random_expectation": {"hit_at_5": round(5 / n, 4), "bottom_error_rate": round(5 / n, 4)},
        "popularity_baseline": popularity,
        "per_town_hit_at_5": {t: round(float(np.mean(by_town[t])), 3) for t in towns},
        "spearman_hit5_vs_reviews": round(float(rho), 4),
    }


# ---------------------------------------------------------------- etiquetas humanas
def human_label_suite(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"sessions": 0}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        return {"sessions": 0}
    precision, bottom_ok, seconds = [], [], []
    by_query: Dict[str, List[Dict[str, int]]] = {}
    for rec in records:
        judgments = rec["judgments"]
        precision.append(np.mean([judgments.get(t, 0) for t in rec["top"]]))
        bottom_ok.append(np.mean([1 - judgments.get(t, 0) for t in rec["bottom"]]))
        if rec.get("seconds"):
            seconds.append(rec["seconds"])
        by_query.setdefault(canonical_text(rec["query"]), []).append(judgments)
    agreements = []
    for judgments_list in by_query.values():
        for i in range(len(judgments_list)):
            for j in range(i + 1, len(judgments_list)):
                a, b = judgments_list[i], judgments_list[j]
                shared = set(a) & set(b)
                if shared:
                    agreements.append(np.mean([a[t] == b[t] for t in shared]))
    return {
        "sessions": len(records),
        "evaluators": len({r.get("evaluator", "anónimo") for r in records}),
        "human_precision_at_5": round(float(np.mean(precision)), 4),
        "human_bottom_unrelated_rate": round(float(np.mean(bottom_ok)), 4),
        "pairwise_agreement": round(float(np.mean(agreements)), 4) if agreements else None,
        "median_seconds_per_query": round(float(np.median(seconds)), 1) if seconds else None,
    }


def rubric(gold: dict, determinism: dict) -> List[dict]:
    agg = gold["aggregate"]
    checks = [
        (
            f"Precision@5 (gold, {agg['wide_queries']} consultas con ≥5 relevantes)",
            agg["precision_at_5_wide"], RUBRIC["precision_at_5"], ">=",
        ),
        ("Hit@1 (gold)", agg["hit_at_1"], RUBRIC["hit_at_1"], ">="),
        ("Contrato 5 + 5", gold["contract_pass_rate"], RUBRIC["contract"], ">="),
        ("Determinismo", determinism["identical_rate"], RUBRIC["determinism"], ">="),
        ("Latencia p95 en proceso (s)", gold["latency_s"]["p95"], RUBRIC["latency_p95_s"], "<"),
    ]
    return [
        {
            "check": name,
            "value": value,
            "threshold": threshold,
            "pass": value >= threshold if op == ">=" else value < threshold,
        }
        for name, value, threshold, op in checks
    ]
