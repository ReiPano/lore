"""Tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from hybrid_search.fusion import DEFAULT_K, reciprocal_rank_fusion


def test_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order() -> None:
    merged = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [doc for doc, _ in merged] == ["a", "b", "c"]


def test_accepts_tuples_and_bare_ids() -> None:
    merged = reciprocal_rank_fusion([[("a", 0.9), ("b", 0.5)], ["b", "a"]])
    ids = [doc for doc, _ in merged]
    assert set(ids) == {"a", "b"}


def test_document_in_both_lists_outranks_singles() -> None:
    bm25 = ["x", "a", "b"]
    vec = ["y", "a", "c"]
    merged = reciprocal_rank_fusion([bm25, vec])
    ranks = {doc: idx for idx, (doc, _) in enumerate(merged)}
    assert ranks["a"] < ranks["x"]
    assert ranks["a"] < ranks["y"]


def test_single_list_hit_still_surfaces() -> None:
    bm25 = ["rare_term_match"]
    vec = ["semantic_a", "semantic_b"]
    merged = reciprocal_rank_fusion([bm25, vec])
    assert "rare_term_match" in {doc for doc, _ in merged}


def test_top_n_truncates() -> None:
    bm25 = [f"b{i}" for i in range(10)]
    vec = [f"v{i}" for i in range(10)]
    merged = reciprocal_rank_fusion([bm25, vec], top_n=5)
    assert len(merged) == 5


def test_weights_bias_results() -> None:
    bm25 = ["a", "b", "c"]
    vec = ["c", "b", "a"]
    # BM25 heavy weighting pushes 'a' up (it was rank 1 in BM25).
    bm25_heavy = reciprocal_rank_fusion([bm25, vec], weights=[10.0, 1.0])
    vec_heavy = reciprocal_rank_fusion([bm25, vec], weights=[1.0, 10.0])
    assert bm25_heavy[0][0] == "a"
    assert vec_heavy[0][0] == "c"


def test_zero_weight_ignores_list() -> None:
    bm25 = ["a", "b"]
    vec = ["c", "d"]
    merged = reciprocal_rank_fusion([bm25, vec], weights=[1.0, 0.0])
    assert [doc for doc, _ in merged] == ["a", "b"]


def test_scores_use_default_k() -> None:
    merged = reciprocal_rank_fusion([["a"]])
    expected = 1.0 / (DEFAULT_K + 1)
    assert merged[0][1] == pytest.approx(expected)


def test_deterministic_tie_break_by_id() -> None:
    # Two docs that only appear at rank 1 of separate lists tie on score;
    # deterministic tie-break on doc_id keeps output stable.
    first = reciprocal_rank_fusion([["z"], ["a"]])
    second = reciprocal_rank_fusion([["z"], ["a"]])
    assert first == second


def test_weights_length_must_match() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_bad_k_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_duplicate_within_single_list_counted_once() -> None:
    # BM25 occasionally repeats an id if the underlying tokenizer creates
    # duplicate postings; the first rank should win, not the last.
    a_rank1 = reciprocal_rank_fusion([["a", "b", "a"]])
    a_only = reciprocal_rank_fusion([["a", "b"]])
    assert a_rank1 == a_only


def test_full_rrf_score_matches_formula() -> None:
    merged = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    scores = dict(merged)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
