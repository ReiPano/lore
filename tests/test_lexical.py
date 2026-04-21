"""Tests for the SQLite FTS5 lexical index."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hybrid_search.chunking import Chunk, make_chunk_id
from hybrid_search.lexical import LexicalIndex


def _chunk(source_path: str, start: int, text: str, **meta) -> Chunk:
    return Chunk(
        id=make_chunk_id(source_path, start, text),
        source_path=source_path,
        text=text,
        start_offset=start,
        end_offset=start + len(text),
        metadata=dict(meta),
    )


@pytest.fixture
def index(tmp_path: Path) -> LexicalIndex:
    idx = LexicalIndex(tmp_path / "lex.sqlite3")
    try:
        yield idx
    finally:
        idx.close()


def test_empty_search_returns_nothing(index: LexicalIndex) -> None:
    assert index.search("anything", k=10) == []
    assert index.search("", k=10) == []
    assert index.count() == 0


def test_add_and_search_recovers_expected_chunk(index: LexicalIndex) -> None:
    docs = [
        _chunk("a.md", 0, "BM25 ranks lexical matches using term frequency and document length."),
        _chunk("b.md", 0, "Dense vector retrieval uses cosine similarity over embeddings."),
        _chunk("c.md", 0, "Reciprocal rank fusion merges heterogeneous result lists."),
    ]
    index.add(docs)
    assert index.count() == 3

    results = index.search("BM25", k=5)
    assert results, "BM25 query must match"
    assert results[0][0] == docs[0].id
    assert all(score > 0 for _, score in results), "scores must be positive after negation"


def test_upsert_replaces_existing(index: LexicalIndex) -> None:
    c = _chunk("a.md", 0, "original text about alpha")
    index.add([c])
    assert index.count() == 1

    updated = Chunk(
        id=c.id,
        source_path=c.source_path,
        text="replacement text about beta",
        start_offset=c.start_offset,
        end_offset=c.end_offset,
        metadata={"revised": True},
    )
    index.add([updated])
    assert index.count() == 1

    assert index.search("alpha") == []
    hits = index.search("beta")
    assert hits and hits[0][0] == c.id
    got = index.get(c.id)
    assert got is not None
    assert got.text == "replacement text about beta"
    assert got.metadata == {"revised": True}


def test_delete_removes_from_both_tables(index: LexicalIndex) -> None:
    c1 = _chunk("a.md", 0, "apples and oranges")
    c2 = _chunk("b.md", 0, "bananas and kiwis")
    index.add([c1, c2])

    deleted = index.delete([c1.id])
    assert deleted == 1
    assert index.count() == 1
    assert index.search("apples") == []
    assert index.search("bananas")[0][0] == c2.id


def test_delete_by_source(index: LexicalIndex) -> None:
    chunks = [
        _chunk("a.md", 0, "alpha one"),
        _chunk("a.md", 20, "alpha two"),
        _chunk("b.md", 0, "beta one"),
    ]
    index.add(chunks)
    removed = index.delete_by_source("a.md")
    assert removed == 2
    assert index.count() == 1
    assert index.sources() == ["b.md"]
    assert index.chunk_ids_for_source("a.md") == []


def test_get_many_preserves_requested_order(index: LexicalIndex) -> None:
    c1 = _chunk("a.md", 0, "first chunk")
    c2 = _chunk("a.md", 100, "second chunk")
    c3 = _chunk("b.md", 0, "third chunk")
    index.add([c1, c2, c3])

    ordered = index.get_many([c3.id, c1.id, "missing", c2.id])
    assert [c.id for c in ordered] == [c3.id, c1.id, c2.id]


def test_query_sanitization_ignores_punctuation(index: LexicalIndex) -> None:
    c = _chunk("a.md", 0, "hybrid search combines BM25 and vectors")
    index.add([c])
    hits = index.search("BM25???!!!", k=5)
    assert hits and hits[0][0] == c.id


def test_large_batch_insert_is_fast(index: LexicalIndex) -> None:
    # 1000 synthetic chunks with a handful of rare markers.
    docs = []
    for i in range(1000):
        text = f"document {i} talks about retrieval strategies and ranking heuristics."
        if i == 777:
            text = "this unique marker xyzzyx identifies the target chunk."
        docs.append(_chunk("synthetic.txt", i * 100, text))

    start = time.monotonic()
    index.add(docs)
    elapsed = time.monotonic() - start
    assert index.count() == 1000
    assert elapsed < 5.0, f"bulk insert slow: {elapsed:.2f}s"

    hits = index.search("xyzzyx", k=10)
    assert hits and hits[0][0] == docs[777].id


def test_scores_are_higher_for_better_matches(index: LexicalIndex) -> None:
    strong = _chunk("a.md", 0, "penguin penguin penguin arctic colony habitat")
    weak = _chunk("b.md", 0, "the penguin was mentioned once in passing")
    other = _chunk("c.md", 0, "unrelated content about kangaroos and koalas")
    index.add([strong, weak, other])

    hits = index.search("penguin", k=5)
    scores = {cid: score for cid, score in hits}
    assert strong.id in scores and weak.id in scores
    assert scores[strong.id] >= scores[weak.id]


def test_reopen_preserves_data(tmp_path: Path) -> None:
    db = tmp_path / "lex.sqlite3"
    with LexicalIndex(db) as idx:
        idx.add([_chunk("a.md", 0, "persistent content about search")])
    with LexicalIndex(db) as idx:
        assert idx.count() == 1
        hits = idx.search("persistent", k=5)
        assert hits
