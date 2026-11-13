#!/usr/bin/env python3
"""API HTTP mínima para conectar el frontend Astro con el motor local.

Arranque en dos fases: el motor léxico responde en segundos; el semántico (torch + E5) se carga en
un hilo y reemplaza al léxico cuando está listo. Cada respuesta declara el modo real en ``mode``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender import Recommender  # noqa: E402
from restmex_recommender import evaluation as ev  # noqa: E402
from restmex_recommender.lab import Lab  # noqa: E402

HOST = os.environ.get("ORBITA_HOST", "127.0.0.1")
PORT = int(os.environ.get("ORBITA_API_PORT", "8765"))
CORS_ORIGIN = os.environ.get("ORBITA_CORS_ORIGIN", "http://127.0.0.1:4321")
ARTIFACTS = Path(os.environ.get("ORBITA_ARTIFACTS", ROOT / "artifacts"))
MODEL_DIR = ROOT / "models/multilingual-e5-base"
GOLD_PATH = ROOT / "eval/gold_queries.json"
LAB_ROOT = Path(os.environ.get("ORBITA_LAB", ROOT / "eval"))
LABELS_PATH = Path(os.environ.get("ORBITA_LABELS", ROOT / "eval/human_labels.jsonl"))
REPORT_PATH = ARTIFACTS / "evaluation_report.json"
MAX_BODY = 200_000  # una sesión del laboratorio puede llevar ~20 juicios con notas

# Secreto por proceso: los tokens de ranking caducan al reiniciar la API (límite documentado).
TOKEN_SECRET = os.urandom(32)
EVAL_LOCK = threading.Lock()
STATE = {"engine": None, "semantic": "pendiente", "started": time.time(), "ready_s": None}
LABELS_LOCK = threading.Lock()


def engine() -> Recommender:
    return STATE["engine"]


def lab() -> Lab:
    """Laboratorio ligado al motor vigente (los pools se amplían al activarse la semántica)."""
    current = engine()
    if STATE.get("lab") is None:
        STATE["lab"] = Lab(current, LAB_ROOT)
    STATE["lab"].engine = current
    return STATE["lab"]


def load_engines() -> None:
    STATE["engine"] = Recommender(ARTIFACTS, enable_semantic=False)
    STATE["ready_s"] = round(time.time() - STATE["started"], 2)
    print(f"[api] motor léxico listo en {STATE['ready_s']} s", flush=True)
    if os.environ.get("ORBITA_SEMANTIC", "1") == "0":
        STATE["semantic"] = "desactivado por ORBITA_SEMANTIC=0"
        return

    def warm() -> None:
        STATE["semantic"] = "cargando"
        try:
            candidate = Recommender(ARTIFACTS, MODEL_DIR)
        except BaseException as exc:  # noqa: BLE001 - el léxico sigue sirviendo
            STATE["semantic"] = f"falló: {type(exc).__name__}: {exc}"[:300]
            print(f"[api] semántica {STATE['semantic']}", flush=True)
            return
        STATE["semantic"] = candidate.semantic_status
        if candidate.semantic_enabled:
            STATE["engine"] = candidate
        print(f"[api] semántica: {candidate.semantic_status}", flush=True)

    threading.Thread(target=warm, name="semantic-warmup", daemon=True).start()


def ranking_token(query: str, mode: str, top: list, bottom: list) -> str:
    """Firma el ranking mostrado; las etiquetas solo se aceptan para rankings emitidos aquí."""
    message = json.dumps([query, mode, top, bottom], ensure_ascii=False).encode("utf-8")
    return hmac.new(TOKEN_SECRET, message, hashlib.sha256).hexdigest()


def _str(payload: dict, key: str, limit: int, default: str = "") -> str:
    value = str(payload.get(key, default)).strip()
    if len(value) > limit:
        raise ValueError(f"El campo {key} excede {limit} caracteres.")
    return value


def validate_labels(payload: dict, towns: set) -> dict:
    """Valida una sesión de evaluación ciega antes de escribirla a disco."""
    query = _str(payload, "query", 500)
    if not query:
        raise ValueError("Falta la consulta evaluada.")
    judgments = payload.get("judgments")
    top, bottom = payload.get("top"), payload.get("bottom")
    if not isinstance(judgments, dict) or not isinstance(top, list) or not isinstance(bottom, list):
        raise ValueError("judgments debe ser objeto; top y bottom, listas.")
    if len(top) != 5 or len(bottom) != 5 or set(top) & set(bottom):
        raise ValueError("Se esperan 5 + 5 destinos distintos.")
    shown = set(top) | set(bottom)
    if not shown <= towns or set(judgments) != shown:
        raise ValueError("Los juicios deben cubrir exactamente los 10 destinos mostrados.")
    if any(value not in (0, 1) for value in judgments.values()):
        raise ValueError("Cada juicio debe ser 0 o 1.")
    expected = ranking_token(query, str(payload.get("mode", "")), top, bottom)
    if not hmac.compare_digest(str(payload.get("token", "")), expected):
        raise ValueError(
            "El ranking no coincide con uno emitido por esta API (¿se reinició el servidor?)."
        )
    seconds = payload.get("seconds")
    if seconds is not None and not (isinstance(seconds, (int, float)) and 0 <= seconds < 86_400):
        raise ValueError("seconds fuera de rango.")
    return {
        "query": query,
        "evaluator": _str(payload, "evaluator", 60, "anónimo") or "anónimo",
        "judgments": {str(k): int(v) for k, v in judgments.items()},
        "top": [str(t) for t in top],
        "bottom": [str(t) for t in bottom],
        "mode": _str(payload, "mode", 60),
        "seconds": seconds,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def run_live_evaluation(current: Recommender) -> dict:
    gold = ev.load_gold(GOLD_PATH)
    gold_result = ev.gold_suite(current, gold)
    determinism = ev.determinism_suite(current, gold)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "híbrido" if current.semantic_enabled else "léxico",
        "gold": gold_result,
        "baselines": ev.baseline_suite(current, gold),
        "adversarial": ev.adversarial_suite(current),
        "determinism": determinism,
        "volume_bias": ev.volume_bias_suite(current, gold_result["queries"]),
        "human": ev.human_label_suite(LABELS_PATH),
        "rubric": ev.rubric(gold_result, determinism),
    }


class ApiHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_text(self, status: int, body: str, content_type: str, filename: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _lab_get(self, route: str, params: dict) -> None:
        if engine() is None:
            self._send(503, {"error": "El motor aún está arrancando."})
            return
        first = {key: values[0] for key, values in params.items()}
        try:
            if route == "/api/lab/queue":
                self._send(200, {"queue": lab().queue(
                    first.get("evaluator", "").strip()[:60] or "anónimo",
                    first.get("split", "dev"),
                )})
            elif route == "/api/lab/task":
                if first.get("query_id") not in lab().by_id:
                    raise ValueError("Consulta desconocida.")
                self._send(200, lab().task(first["query_id"], first.get("evaluator", "anónimo")))
            elif route == "/api/lab/report":
                self._send(200, lab().report())
            elif route == "/api/lab/export":
                fmt = "csv" if first.get("format") == "csv" else "json"
                self._send_text(
                    200, lab().export(fmt),
                    "text/csv; charset=utf-8" if fmt == "csv" else "application/json",
                    f"juicios_humanos.{fmt}",
                )
            else:
                self._send(404, {"error": "Ruta no encontrada."})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        current = engine()
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/lab/"):
            self._lab_get(parsed.path, parse_qs(parsed.query))
            return
        if self.path == "/health":
            self._send(
                200,
                {
                    "status": "ok" if current else "starting",
                    "mode": "híbrido" if current and current.semantic_enabled else "léxico",
                    "semantic": bool(current and current.semantic_enabled),
                    "semantic_status": STATE["semantic"],
                    "lexical_ready_s": STATE["ready_s"],
                    "compat_warnings": current.compat_warnings if current else [],
                    "destinations": len(current.towns) if current else 0,
                },
            )
        elif self.path == "/api/evaluation":
            if not REPORT_PATH.exists():
                self._send(404, {"error": "Aún no hay reporte. Ejecuta make evaluate."})
            else:
                self._send(200, json.loads(REPORT_PATH.read_text(encoding="utf-8")))
        elif self.path == "/api/gold":
            # Modo ciego: solo consultas, nunca las etiquetas esperadas.
            gold = ev.load_gold(GOLD_PATH)
            self._send(200, {"queries": [{"id": q["id"], "query": q["query"]} for q in gold]})
        else:
            self._send(404, {"error": "Ruta no encontrada."})

    def do_POST(self) -> None:  # noqa: N802
        routes = {
            "/api/recommend", "/api/futures", "/api/multiverse", "/api/evaluation/run",
            "/api/labels", "/api/lab/judgments",
        }
        if self.path not in routes:
            self._send(404, {"error": "Ruta no encontrada."})
            return
        current = engine()
        if current is None:
            self._send(503, {"error": "El motor aún está arrancando."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY:
                raise ValueError("La petición está vacía o es demasiado extensa.")
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("El cuerpo debe ser un objeto JSON.")
            if self.path == "/api/evaluation/run":
                if not EVAL_LOCK.acquire(blocking=False):
                    self._send(409, {"error": "Ya hay una evaluación en curso."})
                    return
                try:
                    self._send(200, run_live_evaluation(current))
                finally:
                    EVAL_LOCK.release()
                return
            if self.path == "/api/lab/judgments":
                saved = lab().save(payload)
                self._send(200, {"saved": saved, "report": lab().report()})
                return
            if self.path == "/api/labels":
                record = validate_labels(payload, set(current.towns))
                with LABELS_LOCK:
                    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with LABELS_PATH.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._send(200, {"saved": True, "summary": ev.human_label_suite(LABELS_PATH)})
                return
            query = _str(payload, "query", 2_000)
            risk = float(payload.get("risk", 0.55))
            if self.path == "/api/recommend":
                result = current.recommend(query)
                result["token"] = ranking_token(
                    query,
                    result["mode"],
                    [x["town"] for x in result["top"]],
                    [x["town"] for x in result["bottom"]],
                )
                self._send(200, result)
            elif self.path == "/api/multiverse":
                futures = int(payload.get("futures", 50_000))
                if not 1_000 <= futures <= 100_000:
                    raise ValueError("futures debe estar entre 1,000 y 100,000.")
                choices = payload.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 10:
                    raise ValueError("Las decisiones deben enviarse como una lista (máx. 10).")
                self._send(
                    200,
                    current.explore_multiverse(query, risk=risk, futures=futures, choices=choices),
                )
            else:
                self._send(200, current.recommend_futures(query, risk=risk))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            print(f"[api] error inesperado: {type(exc).__name__}: {exc}", flush=True)
            self._send(500, {"error": "El motor encontró un error inesperado."})

    def log_message(self, message: str, *args: object) -> None:
        print(f"[api] {self.address_string()} — {message % args}", flush=True)


def main() -> None:
    load_engines()
    server = ThreadingHTTPServer((HOST, PORT), ApiHandler)
    print(f"Órbita API disponible en http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo Órbita API…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
