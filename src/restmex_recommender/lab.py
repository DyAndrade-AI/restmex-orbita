"""Laboratorio local de evaluación humana ciega (metodología tipo Cranfield/TREC).

- Banco de consultas con particiones dev / calibration / test (hash estable del id). La
  partición test no se usa para ajustar nada (``tuning_queries`` la excluye por construcción).
- Pool por consulta: unión del top 5 y el bottom 5 de varios sistemas + 3 destinos de la zona
  media (posiciones 11–30) para detectar falsos negativos. Se congela en eval/pools.json; los
  juicios son por (consulta, destino) y sirven para evaluar cualquier sistema futuro.
- Presentación ciega: orden aleatorio por evaluador, sin sistema, posición ni puntuación. La
  información mostrada de cada destino no depende de la consulta.
- Juicio: relevancia 0–3, restricción (cumple / incumple / no_aplica / no_se), evidencia
  insuficiente y nota libre.
- Acuerdo: kappa de Cohen con pesos cuadráticos por pares y alfa de Krippendorff.
- Cola con aprendizaje activo en dev/calibration (prioriza desacuerdo entre sistemas y
  restricciones) mezclada con 30 % de muestreo aleatorio; en test el orden es aleatorio.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .engine import HYBRID_WEIGHTS, LEXICAL_WEIGHTS, Recommender, _rank_order

SPLITS = ("dev", "calibration", "test")
RELEVANCE = (0, 1, 2, 3)
CONSTRAINT = ("cumple", "incumple", "no_aplica", "no_se")
RELEVANCE_LABELS = {
    0: "Nada compatible", 1: "Poco compatible", 2: "Compatible", 3: "Muy compatible",
}
POOL_VERSION = 1
_LOCK = threading.Lock()


def split_for(query_id: str) -> str:
    bucket = int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16) % 4
    return "dev" if bucket < 2 else "calibration" if bucket == 2 else "test"


# ---------------------------------------------------------------- banco de consultas
def load_bank(path: Path) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in data["queries"]:
        item.setdefault("split", split_for(item["id"]))
        if item["split"] not in SPLITS:
            raise ValueError(f"Partición inválida en {item['id']}")
    return data["queries"]


def tuning_queries(bank: Sequence[dict]) -> List[dict]:
    """Consultas permitidas para ajustar el modelo: nunca la partición test."""
    return [q for q in bank if q["split"] != "test"]


# ---------------------------------------------------------------- sistemas y pools
def systems_for(engine: Recommender) -> Dict[str, Optional[dict]]:
    systems: Dict[str, Optional[dict]] = {
        "lexico": dict(LEXICAL_WEIGHTS),
        "tfidf": {"lexical": 1.0},
        "popularidad": None,  # orden por volumen de reseñas
    }
    if engine.bm25 is not None:
        systems["bm25"] = {"bm25": 1.0}
    if engine.semantic_enabled:
        systems["hibrido"] = dict(HYBRID_WEIGHTS)
        systems["e5"] = {"semantic": 1.0}
    return systems


def system_ranking(engine: Recommender, query: str, weights: Optional[dict]) -> List[str]:
    if weights is None:
        counts = np.array([d["review_count"] for d in engine.destinations], dtype=float)
        return [engine.towns[i] for i in _rank_order(counts)]
    scores = engine._score_query(query, weights)["scores"]
    return [engine.towns[i] for i in _rank_order(scores)]


def build_pool(engine: Recommender, item: dict, middle: int = 3) -> dict:
    rankings: Dict[str, List[str]] = {}
    for name, weights in systems_for(engine).items():
        try:
            rankings[name] = system_ranking(engine, item["query"], weights)
        except ValueError:
            continue
    towns: List[str] = []
    for ranking in rankings.values():
        for town in ranking[:5] + ranking[-5:]:
            if town not in towns:
                towns.append(town)
    reference = rankings.get("hibrido") or rankings.get("lexico")
    if reference:
        rng = random.Random(f"middle|{item['id']}")
        candidates = [t for t in reference[10:30] if t not in towns]
        towns += rng.sample(candidates, min(middle, len(candidates)))
    disagreement = 0.0
    tops = [set(r[:5]) for name, r in rankings.items() if name != "popularidad"]
    if len(tops) > 1:
        pairs = [(a, b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        disagreement = 1.0 - float(np.mean([len(a & b) / len(a | b) for a, b in pairs]))
    return {
        "query_id": item["id"],
        "towns": sorted(towns),  # orden canónico; la presentación se baraja por evaluador
        "systems": sorted(rankings),
        "disagreement": round(disagreement, 4),
        "version": POOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class FileStore:
    """Juicios en un JSONL local (desarrollo y Docker con volumen)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def append(self, records: Sequence[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class SupabaseStore:
    """Juicios en Supabase (tabla lab_judgments con una columna jsonb ``record``).

    Para servidores sin disco persistente (Hugging Face Spaces). La clave es secreta y vive solo
    en la API; el navegador nunca habla con Supabase. Se lee todo al arrancar y después se
    mantiene en memoria: la API es un único proceso, así que la caché no se desincroniza."""

    TABLE = "lab_judgments"

    def __init__(self, url: str, key: str, timeout: float = 15.0):
        self.endpoint = f"{url.rstrip('/')}/rest/v1/{self.TABLE}"
        self.headers = {"apikey": key, "Content-Type": "application/json"}
        if key.startswith("eyJ"):  # claves JWT antiguas (service_role) también van como Bearer
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = timeout
        self._cache: Optional[List[dict]] = None

    def _request(self, method: str, url: str, body: Optional[bytes] = None, extra: Optional[dict] = None):
        import urllib.request

        request = urllib.request.Request(url, data=body, method=method,
                                         headers={**self.headers, **(extra or {})})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def load(self) -> List[dict]:
        if self._cache is None:
            rows: List[dict] = []
            page = 1000  # límite por defecto de PostgREST
            while True:
                url = f"{self.endpoint}?select=record&order=id.asc&limit={page}&offset={len(rows)}"
                batch = json.loads(self._request("GET", url))
                rows += [row["record"] for row in batch]
                if len(batch) < page:
                    break
            self._cache = rows
        return list(self._cache)

    def append(self, records: Sequence[dict]) -> None:
        body = json.dumps([{"record": r} for r in records], ensure_ascii=False).encode("utf-8")
        # Si Supabase falla, la excepción sube y la API responde error: el evaluador ve que no se
        # guardó en lugar de perder sus juicios en silencio.
        self._request("POST", self.endpoint, body, {"Prefer": "return=minimal"})
        if self._cache is not None:
            self._cache.extend(records)


def store_from_env(root: Path):
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_KEY", "")
    if url and key:
        return SupabaseStore(url, key)
    return FileStore(Path(root) / "judgments.jsonl")


class Lab:
    def __init__(self, engine: Recommender, root: Path):
        self.engine = engine
        self.root = Path(root)
        self.bank_path = self.root / "query_bank.json"
        self.pools_path = self.root / "pools.json"
        self.store = store_from_env(self.root)
        self.bank = load_bank(self.bank_path)
        self.by_id = {q["id"]: q for q in self.bank}
        self.pools = (
            json.loads(self.pools_path.read_text(encoding="utf-8"))
            if self.pools_path.exists() else {}
        )

    # -- pools
    def pool(self, query_id: str) -> dict:
        with _LOCK:
            if query_id not in self.pools:
                self.pools[query_id] = build_pool(self.engine, self.by_id[query_id])
                self._save_pools()
            else:
                # Extiende con sistemas nuevos sin eliminar destinos ya juzgados.
                fresh = build_pool(self.engine, self.by_id[query_id])
                stored = self.pools[query_id]
                added = [t for t in fresh["towns"] if t not in stored["towns"]]
                if added or set(fresh["systems"]) - set(stored["systems"]):
                    stored["towns"] = sorted(stored["towns"] + added)
                    stored["systems"] = sorted(set(stored["systems"]) | set(fresh["systems"]))
                    self._save_pools()
            return self.pools[query_id]

    def _save_pools(self) -> None:
        self.pools_path.write_text(json.dumps(self.pools, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    # -- presentación ciega
    def neutral_profile(self, town: str) -> dict:
        """Información del destino que NO depende de la consulta (no favorece a ningún sistema)."""
        idx = self.engine.town_to_idx[town]
        item = self.engine.destinations[idx]
        reviews = [self.engine.evidence[i] for i in self.engine.evidence_by_town[town]]
        rng = random.Random(f"neutral|{town}")
        sample = rng.sample(reviews, min(3, len(reviews)))
        return {
            "town": town,
            "name": item["display_name"],
            "region": item["display_region"],
            "topics": item["keywords"]["ALL"][:12],
            "sample_reviews": [
                {"type": r["Type"], "text": str(r["Review"])[:280]} for r in sample
            ],
        }

    def task(self, query_id: str, evaluator: str) -> dict:
        item = self.by_id[query_id]
        pool = self.pool(query_id)
        towns = list(pool["towns"])
        random.Random(f"order|{evaluator}|{query_id}").shuffle(towns)
        return {
            "query_id": query_id,
            "query": item["query"],
            "split": item["split"],
            "has_constraint": bool(self.engine._score_query(item["query"])["intent"].negative)
            if self._answerable(item) else False,
            "destinations": [self.neutral_profile(t) for t in towns],
            "scale": RELEVANCE_LABELS,
        }

    def _answerable(self, item: dict) -> bool:
        try:
            self.engine._score_query(item["query"])
            return True
        except ValueError:
            return False

    # -- cola con aprendizaje activo
    def queue(self, evaluator: str, split: str, limit: int = 10) -> List[dict]:
        if split not in SPLITS:
            raise ValueError("Partición inválida.")
        done = {(j["query_id"]) for j in self.judgments() if j["evaluator"] == evaluator}
        coverage = defaultdict(set)
        for j in self.judgments():
            coverage[j["query_id"]].add(j["evaluator"])
        pending = [q for q in self.bank if q["split"] == split and q["id"] not in done
                   and self._answerable(q)]
        rng = random.Random(f"queue|{evaluator}|{split}")
        if split == "test":
            rng.shuffle(pending)  # sin aprendizaje activo: muestra no sesgada
            return [{"query_id": q["id"], "query": q["query"], "reason": "aleatorio"}
                    for q in pending[:limit]]

        def priority(q: dict) -> float:
            pool = self.pool(q["id"])
            constraint = 0.2 if any(w in q["query"].lower() for w in ("sin ", "no quiero", "evitar")) else 0
            return pool["disagreement"] + constraint - 0.5 * len(coverage[q["id"]])

        by_priority = sorted(pending, key=lambda q: (-priority(q), q["id"]))
        randomized = list(pending)
        rng.shuffle(randomized)
        chosen: List[dict] = []
        taken: set = set()
        while len(chosen) < min(limit, len(pending)):
            # 30 % aleatorio evita que el conjunto juzgado contenga solo casos difíciles.
            source, reason = (
                (randomized, "aleatorio (30 %)") if rng.random() < 0.3 else
                (by_priority, "alta incertidumbre")
            )
            pick = next((q for q in source if q["id"] not in taken), None)
            taken.add(pick["id"])
            chosen.append({"query_id": pick["id"], "query": pick["query"], "reason": reason})
        return chosen

    # -- juicios
    def judgments(self) -> List[dict]:
        rows = self.store.load()
        latest: Dict[tuple, dict] = {}
        for row in rows:  # el último juicio de cada (evaluador, consulta, destino) prevalece
            latest[(row["evaluator"], row["query_id"], row["town"])] = row
        return list(latest.values())

    def save(self, payload: dict) -> int:
        evaluator = str(payload.get("evaluator", "")).strip()[:60]
        query_id = str(payload.get("query_id", ""))
        if not evaluator:
            raise ValueError("Indica un identificador de evaluador.")
        if query_id not in self.by_id:
            raise ValueError("Consulta desconocida.")
        pool = set(self.pool(query_id)["towns"])
        items = payload.get("judgments")
        if not isinstance(items, list) or not items:
            raise ValueError("Faltan juicios.")
        seen = set()
        records = []
        seconds = payload.get("seconds")
        if seconds is not None and not (isinstance(seconds, (int, float)) and 0 <= seconds < 86_400):
            raise ValueError("seconds fuera de rango.")
        now = datetime.now(timezone.utc).isoformat()
        for row in items:
            town = str(row.get("town", ""))
            if town not in pool or town in seen:
                raise ValueError(f"Destino fuera del pool o repetido: {town}")
            relevance = row.get("relevance")
            if relevance not in RELEVANCE:
                raise ValueError("La relevancia debe ser 0, 1, 2 o 3.")
            constraint = row.get("constraint", "no_aplica")
            if constraint not in CONSTRAINT:
                raise ValueError("Valor de restricción inválido.")
            seen.add(town)
            records.append({
                "evaluator": evaluator,
                "query_id": query_id,
                "split": self.by_id[query_id]["split"],
                "town": town,
                "relevance": int(relevance),
                "constraint": constraint,
                "insufficient_evidence": bool(row.get("insufficient_evidence", False)),
                "note": str(row.get("note", ""))[:500],
                "seconds": seconds,
                "pool_version": POOL_VERSION,
                "saved_at": now,
            })
        if seen != pool:
            raise ValueError("Juzga todos los destinos del pool antes de guardar.")
        with _LOCK:
            self.store.append(records)
        return len(records)

    def export(self, fmt: str = "json") -> str:
        rows = self.judgments()
        for row in rows:
            row["query"] = self.by_id.get(row["query_id"], {}).get("query", "")
        if fmt == "json":
            return json.dumps(rows, ensure_ascii=False, indent=1)
        buffer = io.StringIO()
        fields = ["evaluator", "split", "query_id", "query", "town", "relevance", "constraint",
                  "insufficient_evidence", "note", "seconds", "saved_at"]
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    # -- métricas con juicios humanos
    def qrels(self) -> Dict[str, Dict[str, dict]]:
        grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for j in self.judgments():
            grouped[j["query_id"]][j["town"]].append(j)
        out: Dict[str, Dict[str, dict]] = {}
        for qid, towns in grouped.items():
            out[qid] = {}
            for town, rows in towns.items():
                votes = [r["constraint"] for r in rows if r["constraint"] in ("cumple", "incumple")]
                out[qid][town] = {
                    "relevance": float(np.mean([r["relevance"] for r in rows])),
                    "raters": len(rows),
                    "violates": bool(votes) and votes.count("incumple") > len(votes) / 2,
                    "insufficient": sum(r["insufficient_evidence"] for r in rows) > len(rows) / 2,
                }
        return out

    def report(self) -> dict:
        qrels = self.qrels()
        judgments = self.judgments()
        per_system: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for qid, judged in qrels.items():
            item = self.by_id.get(qid)
            if not item or not self._answerable(item):
                continue
            for name, weights in systems_for(self.engine).items():
                try:
                    ranking = system_ranking(self.engine, item["query"], weights)
                except ValueError:
                    continue
                m = human_metrics(ranking, judged)
                for key, value in m.items():
                    if value is not None:
                        per_system[f"{item['split']}"][f"{name}|{key}"].append(value)
        summary = {
            split: _fold(values) for split, values in per_system.items()
        }
        return {
            "judgments": len(judgments),
            "evaluators": len({j["evaluator"] for j in judgments}),
            "queries_judged": len(qrels),
            "by_split_and_system": summary,
            "agreement": agreement(judgments),
            "median_seconds_per_query": _median_seconds(judgments),
            "note": (
                "Métricas calculadas solo sobre destinos juzgados del pool. La partición test "
                "debe consultarse una sola vez al final; no ajustar pesos con ella."
            ),
        }


def _fold(values: Dict[str, list]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for key, items in values.items():
        system, metric = key.split("|")
        out[system][metric] = round(float(np.mean(items)), 4)
        out[system]["queries"] = max(out[system].get("queries", 0), len(items))
    return dict(out)


def _median_seconds(judgments: Sequence[dict]) -> Optional[float]:
    per_session = {(j["evaluator"], j["query_id"]): j["seconds"] for j in judgments if j.get("seconds")}
    return round(float(np.median(list(per_session.values()))), 1) if per_session else None


def human_metrics(ranking: Sequence[str], judged: Dict[str, dict], k: int = 5) -> dict:
    """Métricas graduadas con juicios 0–3 (no juzgados = desconocidos, no irrelevantes)."""
    top, bottom = list(ranking[:k]), list(ranking[-k:])
    rel = {t: v["relevance"] for t, v in judged.items()}
    gains = [rel.get(t, 0.0) for t in top]
    ideal = sorted(rel.values(), reverse=True)[:k]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    judged_top = [t for t in top if t in rel]
    judged_bottom = [t for t in bottom if t in rel]
    middle = [t for t in ranking[k:-k] if t in rel]
    return {
        "ndcg_at_5": dcg / idcg if idcg else None,
        "precision_at_5": (sum(rel[t] >= 2 for t in judged_top) / len(judged_top)
                           if judged_top else None),
        "judged_at_5": len(judged_top) / k,
        "top5_mean_relevance": float(np.mean([rel[t] for t in judged_top])) if judged_top else None,
        "bottom5_mean_relevance": (float(np.mean([rel[t] for t in judged_bottom]))
                                   if judged_bottom else None),
        "bottom5_misplaced": (sum(rel[t] >= 2 for t in judged_bottom) / len(judged_bottom)
                              if judged_bottom else None),
        "top5_constraint_errors": (sum(judged[t]["violates"] for t in judged_top) / len(judged_top)
                                   if judged_top else None),
        "false_negatives_middle": (sum(rel[t] >= 2.5 for t in middle) / len(middle)
                                   if middle else None),
    }


# ---------------------------------------------------------------- acuerdo entre evaluadores
def cohen_kappa_quadratic(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = RELEVANCE
                          ) -> Optional[float]:
    if len(a) != len(b) or len(a) < 2:
        return None
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    denominator = (weights * expected).sum()
    return None if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


def krippendorff_alpha(units: Iterable[Sequence], metric: str = "ordinal") -> Optional[float]:
    """Alfa de Krippendorff (2011) para unidades con ≥2 valores; admite datos faltantes."""
    units = [[v for v in unit if v is not None] for unit in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    values = sorted({v for u in units for v in u}, key=lambda v: (str(type(v)), v))
    index = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for unit in units:
        m = len(unit)
        for i, c in enumerate(unit):
            for j, d in enumerate(unit):
                if i != j:
                    o[index[c], index[d]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return None

    def delta(c: int, d: int) -> float:
        if metric == "nominal":
            return 0.0 if c == d else 1.0
        if metric == "interval":
            return float(values[c] - values[d]) ** 2
        lo, hi = min(c, d), max(c, d)  # ordinal
        return float(n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0) ** 2

    d_matrix = np.array([[delta(c, d) for d in range(k)] for c in range(k)])
    d_o = (o * d_matrix).sum() / n
    d_e = (np.outer(n_c, n_c) * d_matrix).sum() / (n * (n - 1))
    return None if d_e == 0 else float(1.0 - d_o / d_e)


def agreement(judgments: Sequence[dict]) -> dict:
    by_item: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for j in judgments:
        by_item[(j["query_id"], j["town"])][j["evaluator"]] = j
    multi = {key: raters for key, raters in by_item.items() if len(raters) >= 2}
    evaluators = sorted({j["evaluator"] for j in judgments})
    pairs = []
    for i, a in enumerate(evaluators):
        for b in evaluators[i + 1:]:
            shared = [key for key, raters in multi.items() if a in raters and b in raters]
            if len(shared) >= 10:
                kappa = cohen_kappa_quadratic(
                    [multi[key][a]["relevance"] for key in shared],
                    [multi[key][b]["relevance"] for key in shared],
                )
                pairs.append({"a": a, "b": b, "items": len(shared), "kappa_quadratic": kappa})
    relevance_units = [[r["relevance"] for r in raters.values()] for raters in multi.values()]
    constraint_units = [
        [r["constraint"] for r in raters.values() if r["constraint"] in ("cumple", "incumple")]
        for raters in multi.values()
    ]
    return {
        "items_with_2plus_raters": len(multi),
        "krippendorff_alpha_ordinal": krippendorff_alpha(relevance_units, "ordinal"),
        "krippendorff_alpha_interval": krippendorff_alpha(relevance_units, "interval"),
        "constraint_alpha_nominal": krippendorff_alpha(constraint_units, "nominal"),
        "pairwise": pairs,
        "sufficient": len(multi) >= 30,
    }
