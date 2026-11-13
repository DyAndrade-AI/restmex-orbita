#!/usr/bin/env python3
"""Benchmark real de la API: arranque en frío y latencia p50/p95 por HTTP.

    python scripts/benchmark_api.py                 # modo léxico (igual que Docker)
    ORBITA_SEMANTIC=1 python scripts/benchmark_api.py --wait-semantic 600
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET_P95_S = 3.0


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        url, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def start_api(port: int, semantic: bool) -> subprocess.Popen:
    env = {**os.environ, "ORBITA_API_PORT": str(port), "ORBITA_SEMANTIC": "1" if semantic else "0"}
    return subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/serve_api.py")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def wait_health(base: str, timeout: float, predicate=lambda h: h.get("status") == "ok") -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = get(f"{base}/health", timeout=2)
            if predicate(health):
                return health
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"La API no cumplió la condición en {timeout} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--url", help="Mide una API ya en marcha (p. ej. Docker) sin lanzarla")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/api_benchmark.json")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--wait-semantic", type=float, default=0,
                        help="Segundos a esperar a que la semántica termine de cargar")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats debe ser al menos 1")
    semantic = os.environ.get("ORBITA_SEMANTIC", "0") == "1"
    port = free_port()
    base = args.url.rstrip("/") if args.url else f"http://127.0.0.1:{port}"
    queries = [q["query"] for q in json.loads(
        (ROOT / "eval/gold_queries.json").read_text(encoding="utf-8"))["queries"]]

    started = time.perf_counter()
    process = None if args.url else start_api(port, semantic)
    try:
        health = wait_health(base, args.startup_timeout)
        cold_start = time.perf_counter() - started
        semantic_ready, semantic_outcome = None, "no solicitada"
        if semantic and args.wait_semantic:
            # Terminal = activo o cualquier estado que no sea pendiente/cargando (fallo/fallback).
            def terminal(h: dict) -> bool:
                return h["semantic"] or h["semantic_status"] not in ("pendiente", "cargando")

            try:
                health = wait_health(base, args.wait_semantic, terminal)
                if health["semantic"]:
                    semantic_ready = round(time.perf_counter() - started, 2)
                    semantic_outcome = "activa"
                else:
                    semantic_outcome = f"fallida: {health['semantic_status']}"
            except TimeoutError:
                health = get(f"{base}/health")
                semantic_outcome = f"timeout tras {args.wait_semantic} s"
        first = time.perf_counter()
        post(f"{base}/api/recommend", {"query": queries[0]})
        first_query = time.perf_counter() - first

        latencies, modes = [], set()
        for _ in range(args.repeats):
            for query in queries:
                t0 = time.perf_counter()
                result = post(f"{base}/api/recommend", {"query": query})
                latencies.append(time.perf_counter() - t0)
                modes.add(result["mode"])
        p95 = float(np.percentile(latencies, 95))
        report = {
            "target": base if args.url else "proceso local lanzado por el benchmark",
            "requests": len(latencies),
            "mode_reported_by_responses": sorted(modes),
            "health": health,
            "cold_start_to_health_s": round(cold_start, 2),
            "semantic_ready_s": semantic_ready,
            "semantic_outcome": semantic_outcome,
            "first_query_s": round(first_query, 3),
            "latency_s": {
                "p50": round(float(np.percentile(latencies, 50)), 3),
                "p95": round(p95, 3),
                "max": round(max(latencies), 3),
            },
            "target_p95_s": TARGET_P95_S,
            "p95_pass": p95 < TARGET_P95_S,
        }
    finally:
        if process:
            process.terminate()
            process.wait(timeout=10)
    if args.url:
        report["cold_start_to_health_s"] = None  # se mide aparte al arrancar el contenedor
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["p95_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
