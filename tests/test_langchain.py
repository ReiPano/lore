"""Tests for the LangChain retriever integration."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langchain_core")

import httpx

from hybrid_search.integrations.langchain import DEFAULT_BASE_URL, LoreRetriever


def _make_transport(expected_payload: dict, hits: list[dict]) -> httpx.MockTransport:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"results": hits})

    captured["expected"] = expected_payload
    return httpx.MockTransport(handler), captured


@pytest.fixture
def sample_hits() -> list[dict]:
    return [
        {
            "chunk_id": "abc123",
            "text": "first hit body",
            "source_path": "/abs/proj/a.md",
            "score": 0.87,
            "scores_breakdown": {"bm25": 0.5, "vector": 0.42, "rrf": 0.87},
        },
        {
            "chunk_id": "def456",
            "text": "second hit body",
            "source_path": "/abs/proj/b.md",
            "score": 0.55,
            "scores_breakdown": {"rrf": 0.55},
        },
    ]


def test_retriever_posts_expected_payload(monkeypatch, sample_hits) -> None:
    transport, captured = _make_transport({}, sample_hits)
    real_client = httpx.Client

    def patched(timeout):  # noqa: ANN001
        return real_client(transport=transport, timeout=timeout)

    monkeypatch.setattr(httpx, "Client", patched)

    retriever = LoreRetriever(k=7, rerank=True, project="proj-a")
    docs = retriever.invoke("needle")

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/search")
    assert captured["body"] == {
        "query": "needle",
        "k": 7,
        "rerank": True,
        "project": "proj-a",
    }
    assert len(docs) == 2
    assert docs[0].page_content == "first hit body"
    assert docs[0].metadata["chunk_id"] == "abc123"
    assert docs[0].metadata["source_path"] == "/abs/proj/a.md"
    assert docs[0].metadata["scores_breakdown"] == {
        "bm25": 0.5,
        "vector": 0.42,
        "rrf": 0.87,
    }


def test_retriever_omits_optional_fields_when_none(monkeypatch, sample_hits) -> None:
    transport, captured = _make_transport({}, sample_hits)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: real_client(transport=transport, timeout=timeout),
    )

    retriever = LoreRetriever(base_url=DEFAULT_BASE_URL)
    retriever.invoke("x")
    body = captured["body"]
    assert "rerank" not in body
    assert "project" not in body
    assert body["k"] == 10


def test_retriever_auth_header(monkeypatch, sample_hits) -> None:
    transport, captured = _make_transport({}, sample_hits)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: real_client(transport=transport, timeout=timeout),
    )

    retriever = LoreRetriever(auth_token="s3cr3t")
    retriever.invoke("q")
    assert captured["headers"]["authorization"] == "Bearer s3cr3t"


def test_retriever_async(monkeypatch, sample_hits) -> None:
    transport, captured = _make_transport({}, sample_hits)
    real_async = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda timeout: real_async(transport=transport, timeout=timeout),
    )

    retriever = LoreRetriever()
    docs = asyncio.run(retriever.ainvoke("needle"))
    assert captured["body"]["query"] == "needle"
    assert len(docs) == 2


def test_retriever_raises_on_http_error(monkeypatch) -> None:
    def failure_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    transport = httpx.MockTransport(failure_handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda timeout: real_client(transport=transport, timeout=timeout),
    )

    retriever = LoreRetriever()
    with pytest.raises(httpx.HTTPStatusError):
        retriever.invoke("q")
