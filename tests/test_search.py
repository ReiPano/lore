"""End-to-end HybridSearch tests using in-memory Qdrant and fake embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_search.chunking import Chunk, make_chunk_id
from hybrid_search.lexical import LexicalIndex
from hybrid_search.search import HybridSearch, SearchResult
from hybrid_search.vector import VectorIndex

from .conftest import FakeEmbedder


class KeywordReranker:
    """Reranker that scores by keyword count — handy for tests."""

    model_name = "keyword-rerank-v1"

    def __init__(self, preferred: str) -> None:
        self.preferred = preferred.lower()
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return [float(t.lower().count(self.preferred)) for t in texts]


def _chunk(source: str, start: int, text: str) -> Chunk:
    return Chunk(
        id=make_chunk_id(source, start, text),
        source_path=source,
        text=text,
        start_offset=start,
        end_offset=start + len(text),
    )


@pytest.fixture
def fixture_corpus() -> list[Chunk]:
    return [
        _chunk(
            "docs/search.md",
            0,
            "Hybrid search combines BM25 and vector retrieval for better recall.",
        ),
        _chunk(
            "docs/bm25.md",
            0,
            "BM25 ranks lexical matches using term frequency and document length.",
        ),
        _chunk(
            "docs/vectors.md",
            0,
            "Dense vector retrieval uses cosine similarity over embeddings.",
        ),
        _chunk(
            "docs/rrf.md",
            0,
            "Reciprocal rank fusion merges heterogeneous result lists without calibrated scores.",
        ),
        _chunk(
            "docs/rerank.md",
            0,
            "Cross-encoder rerankers polish top candidates by jointly scoring query document pairs.",
        ),
    ]


@pytest.fixture
def stack(tmp_path: Path, fixture_corpus: list[Chunk]):
    lex = LexicalIndex(tmp_path / "lex.sqlite3")
    vec = VectorIndex.in_memory("hybrid_test", FakeEmbedder(dim=32))
    lex.add(fixture_corpus)
    vec.add(fixture_corpus)
    yield lex, vec, fixture_corpus
    lex.close()
    vec.close()


def test_query_returns_expected_chunk_without_rerank(stack) -> None:
    lex, vec, corpus = stack
    search = HybridSearch(lex, vec, rerank_enabled=False, retrieval_k_per_index=10, default_k=3)
    results = search.query("BM25 ranks lexical matches using term frequency and document length.")
    assert results
    assert results[0].chunk_id == corpus[1].id
    assert isinstance(results[0], SearchResult)


def test_score_breakdown_contains_components(stack) -> None:
    lex, vec, corpus = stack
    search = HybridSearch(lex, vec, rerank_enabled=False, retrieval_k_per_index=10, default_k=3)
    # Exact-text query guarantees both BM25 and vector hit the same chunk.
    results = search.query(corpus[0].text)
    top = results[0]
    assert "bm25" in top.scores_breakdown
    assert "vector" in top.scores_breakdown
    assert "rrf" in top.scores_breakdown
    assert "rerank" not in top.scores_breakdown
    assert top.score == pytest.approx(top.scores_breakdown["rrf"])


def test_rerank_changes_top_result(stack) -> None:
    lex, vec, corpus = stack
    rr = KeywordReranker(preferred="cross-encoder")
    search = HybridSearch(
        lex,
        vec,
        reranker=rr,
        rerank_enabled=True,
        retrieval_k_per_index=10,
        rerank_top_n=5,
        default_k=3,
    )
    results = search.query("retrieval quality")
    assert rr.calls >= 1
    assert results[0].chunk_id == corpus[4].id  # rerank.md preferred by keyword count
    assert "rerank" in results[0].scores_breakdown


def test_rerank_flag_disables_reranker_at_call_time(stack) -> None:
    lex, vec, corpus = stack
    rr = KeywordReranker(preferred="cross-encoder")
    search = HybridSearch(lex, vec, reranker=rr, rerank_enabled=True, retrieval_k_per_index=10, default_k=3)
    search.query(corpus[0].text, rerank=False)
    assert rr.calls == 0


def test_empty_query_returns_empty(stack) -> None:
    lex, vec, _ = stack
    search = HybridSearch(lex, vec, rerank_enabled=False)
    assert search.query("") == []
    assert search.query("   ") == []
    assert search.query("anything", k=0) == []


def test_async_and_sync_agree(stack) -> None:
    import asyncio

    lex, vec, corpus = stack
    search = HybridSearch(lex, vec, rerank_enabled=False, retrieval_k_per_index=10, default_k=3)
    sync_results = search.query(corpus[0].text)
    async_results = asyncio.run(search.aquery(corpus[0].text))
    assert [r.chunk_id for r in sync_results] == [r.chunk_id for r in async_results]


def test_missing_hydration_is_skipped(stack) -> None:
    """If a fused hit is deleted from SQLite before hydration, it drops cleanly."""

    lex, vec, corpus = stack
    victim = corpus[2]
    # Drop from lexical but leave vector index unchanged — simulates a race.
    lex.delete([victim.id])
    search = HybridSearch(lex, vec, rerank_enabled=False, retrieval_k_per_index=10, default_k=5)
    results = search.query(victim.text)
    assert all(r.chunk_id != victim.id for r in results)


def test_returned_results_respect_k(stack) -> None:
    lex, vec, _ = stack
    search = HybridSearch(lex, vec, rerank_enabled=False, retrieval_k_per_index=10)
    assert len(search.query("vector", k=2)) <= 2
    assert len(search.query("vector", k=10)) <= 5  # corpus has 5 chunks
