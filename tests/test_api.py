"""FastAPI HTTP API tests using the built-in TestClient."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hybrid_search.api import create_app
from hybrid_search.indexer import Indexer
from hybrid_search.lexical import LexicalIndex
from hybrid_search.search import HybridSearch
from hybrid_search.vector import VectorIndex

from .conftest import FakeEmbedder


@pytest.fixture
def stack(tmp_path: Path):
    lex = LexicalIndex(tmp_path / "lex.sqlite3")
    vec = VectorIndex.in_memory("api_test", FakeEmbedder(dim=32))
    indexer = Indexer(lex, vec, chunk_size=500, chunk_overlap=50)
    search = HybridSearch(
        lex,
        vec,
        rerank_enabled=False,
        retrieval_k_per_index=10,
        default_k=5,
    )
    yield lex, vec, indexer, search, tmp_path
    lex.close()
    vec.close()


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _client(indexer: Indexer, search: HybridSearch, *, auth_token: str | None = None) -> TestClient:
    app = create_app(search=search, indexer=indexer, auth_token=auth_token)
    return TestClient(app)


def _poll_job(client: TestClient, job_id: str, *, timeout: float = 5.0, headers=None) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/jobs/{job_id}", headers=headers or {})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in {"done", "error"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_health_empty(stack) -> None:
    _, _, indexer, search, _ = stack
    with _client(indexer, search) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunk_count"] == 0
        assert body["sources_count"] == 0
        assert body["embedding_model"] == "fake-hash-v1"
        assert body["collection_name"] == "api_test"


def test_index_job_indexes_and_reports_stats(stack) -> None:
    _, _, indexer, search, tmp = stack
    _write(tmp, "a.md", "# Alpha\n\nContent about search.")
    _write(tmp, "b.md", "# Beta\n\nMore content here.")
    with _client(indexer, search) as client:
        resp = client.post("/index", json={"paths": [str(tmp)]})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        job = _poll_job(client, job_id)
        assert job["status"] == "done"
        stats = job["stats"]
        assert stats["files_indexed"] == 2
        assert stats["chunks_added"] > 0


def test_search_returns_indexed_content(stack) -> None:
    lex, vec, indexer, search, tmp = stack
    _write(tmp, "search.md", "# Hybrid\n\nBM25 combined with vectors is hybrid search.")
    indexer.index_path(tmp)
    with _client(indexer, search) as client:
        resp = client.post(
            "/search",
            json={"query": "BM25 combined with vectors is hybrid search.", "k": 3},
        )
        assert resp.status_code == 200, resp.text
        results = resp.json()["results"]
        assert results
        assert "search.md" in results[0]["source_path"]


def test_search_requires_query(stack) -> None:
    _, _, indexer, search, _ = stack
    with _client(indexer, search) as client:
        resp = client.post("/search", json={"query": "", "k": 3})
        assert resp.status_code == 422


def test_delete_by_paths(stack) -> None:
    lex, _, indexer, search, tmp = stack
    target = _write(tmp, "drop.md", "# Drop\n\nThis will be removed.")
    indexer.index_file(target)
    assert lex.count() > 0
    with _client(indexer, search) as client:
        resp = client.request("DELETE", "/documents", json={"paths": [str(target)]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["files_removed"] == 1
        assert body["chunks_removed"] > 0
        assert lex.count() == 0


def test_delete_by_chunk_ids(stack) -> None:
    lex, _, indexer, search, tmp = stack
    target = _write(tmp, "doc.md", "# Doc\n\nContent lives on.")
    indexer.index_file(target)
    ids = lex.chunk_ids_for_source(str(target.resolve()))
    assert ids
    with _client(indexer, search) as client:
        resp = client.request("DELETE", "/documents", json={"chunk_ids": ids})
        assert resp.status_code == 200
        assert resp.json()["chunks_removed"] == len(ids)
        assert lex.count() == 0


def test_delete_requires_targets(stack) -> None:
    _, _, indexer, search, _ = stack
    with _client(indexer, search) as client:
        resp = client.request("DELETE", "/documents", json={})
        assert resp.status_code == 400


def test_missing_job_returns_404(stack) -> None:
    _, _, indexer, search, _ = stack
    with _client(indexer, search) as client:
        resp = client.get("/jobs/does-not-exist")
        assert resp.status_code == 404


def test_auth_required_when_token_set(stack) -> None:
    _, _, indexer, search, tmp = stack
    _write(tmp, "a.md", "# Alpha\n\nContent.")
    with _client(indexer, search, auth_token="secret") as client:
        resp = client.get("/health")
        assert resp.status_code == 401

        resp = client.get("/health", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        resp = client.get("/health", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200


def test_index_job_error_surfaces(stack) -> None:
    _, _, indexer, search, _ = stack
    with _client(indexer, search) as client:
        resp = client.post("/index", json={"paths": ["/definitely/does/not/exist"]})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        job = _poll_job(client, job_id)
        # index_path records the missing path as a stats error but does not raise;
        # the job completes successfully with that entry captured.
        assert job["status"] == "done"
        assert job["stats"]["errors"]
