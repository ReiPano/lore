"""Tests for the eval metrics + smoke run of the pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eval.metrics import (
    Aggregate,
    QueryResult,
    aggregate,
    hit_at_k,
    is_relevant,
    mrr,
    ndcg_at_k,
)
from eval.run_eval import load_queries, run_eval

from .conftest import FakeEmbedder


def test_is_relevant_case_insensitive() -> None:
    assert is_relevant("BM25 ranks", ["bm25"])
    assert is_relevant("deep BM25 content", ["rare_term", "bm25"])
    assert not is_relevant("unrelated", ["bm25"])
    assert not is_relevant("anything", [])


def test_hit_at_k_and_mrr() -> None:
    assert hit_at_k([3, 7], 10) == 1.0
    assert hit_at_k([12], 10) == 0.0
    assert mrr([5, 2, 9]) == pytest.approx(1 / 2)
    assert mrr([]) == 0.0


def test_ndcg_at_k_basic() -> None:
    # One relevant at rank 1 → ndcg = 1.
    assert ndcg_at_k([1], 10) == pytest.approx(1.0)
    # Relevant past cutoff → zero.
    assert ndcg_at_k([20], 10) == pytest.approx(0.0)
    # Two hits, first deeper → DCG of 1/log2(3) + 1/log2(5); idcg=1.
    import math

    expected = (1.0 / math.log2(3) + 1.0 / math.log2(5)) / 1.0
    assert ndcg_at_k([2, 4], 10) == pytest.approx(expected)


def test_aggregate_handles_empty() -> None:
    agg = aggregate([])
    assert isinstance(agg, Aggregate)
    assert agg.n_queries == 0


def test_aggregate_computes_means() -> None:
    results = [
        QueryResult(
            query="q1",
            retrieved_texts=["bm25 rank", "unrelated", "more"],
            must_contain=["bm25"],
        ),
        QueryResult(
            query="q2",
            retrieved_texts=["nothing here", "still nothing"],
            must_contain=["missing"],
        ),
    ]
    agg = aggregate(results, k=10)
    # q1: ranks=[1]; q2: ranks=[]
    assert agg.hit_at_10 == pytest.approx(0.5)
    assert agg.mrr == pytest.approx(0.5)  # (1 + 0) / 2
    assert agg.n_queries == 2


def test_load_queries(tmp_path: Path) -> None:
    jl = tmp_path / "q.jsonl"
    jl.write_text(
        '{"query": "a", "must_contain": ["x"]}\n'
        "\n"
        '{"query": "b", "must_contain": ["y"], "notes": "hi"}\n',
        encoding="utf-8",
    )
    qs = load_queries(jl)
    assert len(qs) == 2
    assert qs[0].query == "a"
    assert qs[1].notes == "hi"


def test_run_eval_smoke(tmp_path: Path) -> None:
    # Tiny corpus: query text appears verbatim in one doc so even the fake
    # embedder's exact-match lookup pipes through.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# A\n\nChunk talks about the target phrase needle_x directly.",
        encoding="utf-8",
    )
    (corpus / "b.md").write_text(
        "# B\n\nUnrelated filler content, different topic entirely.",
        encoding="utf-8",
    )

    from eval.run_eval import EvalQuery

    queries = [
        EvalQuery(
            query="Chunk talks about the target phrase needle_x directly.",
            must_contain=["needle_x"],
        ),
    ]
    results = asyncio.run(
        run_eval(
            corpus,
            queries,
            embedder=FakeEmbedder(dim=32),
            include_rerank=False,
            k=5,
        )
    )
    assert set(results) == {"bm25", "vector", "hybrid"}
    for name, (agg, _ms) in results.items():
        assert agg.n_queries == 1
        assert 0.0 <= agg.hit_at_10 <= 1.0
        assert 0.0 <= agg.mrr <= 1.0
        assert 0.0 <= agg.ndcg_at_10 <= 1.0
