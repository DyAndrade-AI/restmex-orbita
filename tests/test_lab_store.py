"""Almacenamiento de juicios: archivo local y Supabase (contra un PostgREST falso)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from restmex_recommender.lab import FileStore, SupabaseStore, store_from_env


class FakePostgrest(BaseHTTPRequestHandler):
    rows: list = []
    seen_headers: list = []
    fail_next_post = False

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        type(self).seen_headers.append({k.lower(): v for k, v in self.headers.items()})
        q = parse_qs(urlparse(self.path).query)
        limit, offset = int(q["limit"][0]), int(q["offset"][0])
        body = json.dumps(type(self).rows[offset:offset + limit]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        type(self).seen_headers.append({k.lower(): v for k, v in self.headers.items()})
        if type(self).fail_next_post:
            type(self).fail_next_post = False
            self.send_response(500)
            self.end_headers()
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).rows += data
        self.send_response(201)
        self.end_headers()


@pytest.fixture()
def postgrest():
    FakePostgrest.rows, FakePostgrest.seen_headers = [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakePostgrest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_supabase_roundtrip_and_pagination(postgrest):
    FakePostgrest.rows = [{"record": {"n": i}} for i in range(2500)]  # > 2 páginas de 1000
    store = SupabaseStore(postgrest, "sb_secret_x")
    assert [r["n"] for r in store.load()] == list(range(2500))
    store.append([{"n": "a"}, {"n": "b"}])
    assert store.load()[-2:] == [{"n": "a"}, {"n": "b"}]  # caché actualizada
    assert SupabaseStore(postgrest, "sb_secret_x").load()[-1] == {"n": "b"}  # persistido
    assert FakePostgrest.seen_headers[0]["apikey"] == "sb_secret_x"
    assert "authorization" not in FakePostgrest.seen_headers[0]


def test_jwt_key_also_sent_as_bearer(postgrest):
    SupabaseStore(postgrest, "eyJabc").load()
    assert FakePostgrest.seen_headers[0]["authorization"] == "Bearer eyJabc"


def test_failed_write_raises_and_keeps_cache_clean(postgrest):
    store = SupabaseStore(postgrest, "k")
    store.load()
    FakePostgrest.fail_next_post = True
    with pytest.raises(Exception):
        store.append([{"n": 1}])
    assert store.load() == []


def test_store_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    assert isinstance(store_from_env(tmp_path), FileStore)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    assert isinstance(store_from_env(tmp_path), SupabaseStore)


def test_file_store_roundtrip(tmp_path):
    store = FileStore(tmp_path / "sub" / "j.jsonl")
    assert store.load() == []
    store.append([{"a": 1}, {"a": 2}])
    assert store.load() == [{"a": 1}, {"a": 2}]
