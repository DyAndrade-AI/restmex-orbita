#!/usr/bin/env python3
"""Smoke test end-to-end: API + Astro (dev con proxy) + accesibilidad básica del HTML.

Arranca ambos procesos, recorre el flujo del evaluador (recomendar → etiquetar) a través del
proxy de Astro y revisa reglas estáticas de accesibilidad. Sale con código 1 si algo falla.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_api import get, post, wait_health  # noqa: E402

WEB = "http://127.0.0.1:4321"


class A11yParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.issues: list = []
        self.lang = None
        self.labels_for: set = set()
        self.fields: list = []
        self.headings: list = []
        self.h1 = 0
        self._button: dict = {}
        self._title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        elif tag == "title":
            self._title = True
        elif tag == "label" and a.get("for"):
            self.labels_for.add(a["for"])
        elif tag in ("input", "textarea", "select") and a.get("type") != "hidden":
            self.fields.append(a)
        elif tag == "img" and "alt" not in a:
            self.issues.append(f"<img> sin alt: {a.get('src')}")
        elif tag == "button":
            self._button = {"attrs": a, "text": ""}
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            if self.headings and level > self.headings[-1] + 1:
                self.issues.append(f"salto de encabezado h{self.headings[-1]} → h{level}")
            self.headings.append(level)
            self.h1 += level == 1

    def handle_data(self, data):
        if self._button:
            self._button["text"] += data.strip()
        if self._title:
            self.title += data.strip()

    def handle_endtag(self, tag):
        if tag == "title":
            self._title = False
        if tag == "button" and self._button:
            attrs = self._button["attrs"]
            if not (self._button["text"] or attrs.get("aria-label")):
                self.issues.append("<button> sin nombre accesible")
            self._button = {}

    def report(self) -> list:
        issues = list(self.issues)
        if not self.lang:
            issues.append("<html> sin lang")
        if not self.title:
            issues.append("documento sin <title>")
        if self.h1 != 1:
            issues.append(f"se esperaba un h1, hay {self.h1}")
        for field in self.fields:
            named = field.get("id") in self.labels_for or field.get("aria-label")
            if not named:
                issues.append(f"campo sin etiqueta: {field.get('id') or field.get('name')}")
        return issues


def main() -> None:
    failures: list = []
    scratch = Path(tempfile.mkdtemp())
    labels = scratch / "labels.jsonl"
    lab_dir = scratch / "lab"
    lab_dir.mkdir()
    (lab_dir / "query_bank.json").write_text(
        (ROOT / "eval/query_bank.json").read_text(encoding="utf-8"), encoding="utf-8")
    env = {**os.environ, "ORBITA_SEMANTIC": "0", "ORBITA_LABELS": str(labels),
           "ORBITA_LAB": str(lab_dir)}
    api = subprocess.Popen([sys.executable, str(ROOT / "scripts/serve_api.py")], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    web = subprocess.Popen(["npm", "run", "dev", "--", "--port", "4321"], cwd=ROOT / "web",
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        health = wait_health("http://127.0.0.1:8765", 600)
        print(f"API lista · modo {health['mode']} · {health['semantic_status']}")
        deadline = time.time() + 120
        html = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(WEB + "/", timeout=5) as response:
                    html = response.read().decode("utf-8")
                break
            except OSError:
                time.sleep(0.5)
        if not html:
            raise TimeoutError("Astro no respondió en 120 s")

        for marker in ('id="classic-form"', 'id="eval-panel"', 'id="lab-panel"'):
            if marker not in html:
                failures.append(f"falta {marker} en la página")
        parser = A11yParser()
        parser.feed(html)
        failures += [f"a11y: {issue}" for issue in parser.report()]

        # Flujo del evaluador a través del proxy de Astro.
        query = "Quiero cenotes para nadar y comida yucateca"
        result = post(WEB + "/api/recommend", {"query": query})
        top = [x["town"] for x in result["top"]]
        bottom = [x["town"] for x in result["bottom"]]
        if len(top) != 5 or len(bottom) != 5 or set(top) & set(bottom):
            failures.append("contrato 5 + 5 roto vía proxy")
        if not get(WEB + "/api/gold")["queries"]:
            failures.append("/api/gold vacío")
        label_payload = {
            "query": query, "mode": result["mode"], "top": top, "bottom": bottom,
            "token": result["token"], "evaluator": "smoke", "seconds": 12,
            "judgments": {**{t: 1 for t in top}, **{t: 0 for t in bottom}},
        }
        saved = post(WEB + "/api/labels", label_payload)
        if saved["summary"]["sessions"] != 1:
            failures.append("la etiqueta no se guardó")
        try:
            # Payload válido salvo el token: el rechazo debe venir de la firma HMAC.
            post(WEB + "/api/labels", {**label_payload, "token": "falso"})
            failures.append("se aceptó una etiqueta con token falso")
        except urllib.error.HTTPError as exc:
            message = json.load(exc).get("error", "")
            if exc.code != 400 or "no coincide" not in message:
                failures.append(f"token falso: {exc.code} {message}")
        # Laboratorio: cola → tarea ciega → juicios 0–3 → reporte → exportación.
        queue = get(WEB + "/api/lab/queue?evaluator=smoke&split=dev")["queue"]
        task = get(WEB + f"/api/lab/task?query_id={queue[0]['query_id']}&evaluator=smoke")
        if "score" in json.dumps(task):
            failures.append("la tarea ciega expone puntuaciones")
        saved_lab = post(WEB + "/api/lab/judgments", {
            "evaluator": "smoke", "query_id": task["query_id"], "seconds": 30,
            "judgments": [{"town": d["town"], "relevance": i % 4, "constraint": "no_aplica"}
                          for i, d in enumerate(task["destinations"])],
        })
        if saved_lab["saved"] != len(task["destinations"]):
            failures.append("el laboratorio no guardó todos los juicios")
        with urllib.request.urlopen(WEB + "/api/lab/export?format=csv", timeout=10) as response:
            if "relevance" not in response.read().decode("utf-8").splitlines()[0]:
                failures.append("exportación CSV sin encabezado esperado")
        orbit = post(WEB + "/api/futures", {"query": query})
        if len(orbit["destinations"]) != 5 or len(orbit["bottom_perspective"]) != 5:
            failures.append("Órbita incompleta")
        multiverse = post(WEB + "/api/multiverse", {"query": query, "futures": 20000})
        if "monte_carlo" not in multiverse or len(multiverse["bottom_in_multiverse"]) != 5:
            failures.append("Strange Loop sin incertidumbre o sin análisis del bottom")
        evaluation = post(WEB + "/api/evaluation/run", {}, timeout=600)
        print("Rúbrica:", json.dumps(evaluation["rubric"], ensure_ascii=False))
    finally:
        for process in (web, api):
            process.terminate()
            process.wait(timeout=15)

    if failures:
        print("FALLOS:\n- " + "\n- ".join(failures))
        sys.exit(1)
    print("Smoke E2E OK: API, proxy Astro, flujo de etiquetas y accesibilidad básica.")


if __name__ == "__main__":
    main()
