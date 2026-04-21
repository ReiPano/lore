"""Tests for the Qdrant vector index using the in-memory client."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from hybrid_search.chunking import Chunk, make_chunk_id
from hybrid_search.vector import (
    META_POINT_ID,
    VectorIndex,
    VectorIndexError,
    _chunk_point_id,
)

from .conftest import FakeEmbedder


def _chunk(source_path: str, start: int, text: str) -> Chunk:
    return Chunk(
        id=make_chunk_id(source_path, start, text),
        source_path=source_path,
        text=text,
        start_offset=start,
        end_offset=start + len(text),
    )


@pytest.fixture
def index() -> VectorIndex:
    idx = VectorIndex.in_memory("test_collection", FakeEmbedder(dim=32))
    try:
        yield idx
    finally:
        idx.close()


def test_chunk_point_id_is_stable_and_non_zero() -> None:
    cid = make_chunk_id("a.md", 0, "hello world")
    assert _chunk_point_id(cid) == _chunk_point_id(cid)
    assert _chunk_point_id(cid) != META_POINT_ID
    assert _chunk_point_id("0" * 16) == 1


def test_count_empty(index: VectorIndex) -> None:
    assert index.count() == 0
    assert index.search("anything", k=5) == []


def test_add_and_search_round_trip(index: VectorIndex) -> None:
    chunks = [
        _chunk("a.md", 0, "alpha one"),
        _chunk("b.md", 0, "beta two"),
        _chunk("c.md", 0, "gamma three"),
    ]
    assert index.add(chunks) == 3
    assert index.count() == 3

    hits = index.search("alpha one", k=5)
    assert hits, "hash-based embedder must recover exact text match"
    assert hits[0][0] == chunks[0].id
    assert all(isinstance(score, float) for _, score in hits)


def test_search_does_not_return_meta_point(index: VectorIndex) -> None:
    chunks = [_chunk("a.md", 0, "some content")]
    index.add(chunks)
    hits = index.search("totally unrelated phrase", k=10)
    # Nothing in the results payload should be the meta sentinel.
    assert all(cid == chunks[0].id for cid, _ in hits)


def test_delete_by_id_removes_vector(index: VectorIndex) -> None:
    a = _chunk("a.md", 0, "keepable content")
    b = _chunk("b.md", 0, "removable content")
    index.add([a, b])
    assert index.delete([b.id]) == 1
    assert index.count() == 1
    hits = index.search("removable content", k=5)
    assert all(cid != b.id for cid, _ in hits)


def test_delete_by_source(index: VectorIndex) -> None:
    chunks = [
        _chunk("a.md", 0, "first a"),
        _chunk("a.md", 100, "second a"),
        _chunk("b.md", 0, "only b"),
    ]
    index.add(chunks)
    index.delete_by_source("a.md")
    assert index.count() == 1
    hits = index.search("only b", k=5)
    assert hits and hits[0][0] == chunks[2].id


def test_upsert_is_idempotent(index: VectorIndex) -> None:
    c = _chunk("a.md", 0, "text")
    index.add([c])
    index.add([c])
    assert index.count() == 1


def test_search_with_empty_query_returns_empty(index: VectorIndex) -> None:
    index.add([_chunk("a.md", 0, "anything")])
    assert index.search("", k=5) == []
    assert index.search("   ", k=5) == []
    assert index.search("valid", k=0) == []


def test_model_mismatch_raises() -> None:
    client = QdrantClient(":memory:")
    try:
        VectorIndex(client, "coll", FakeEmbedder(dim=16))
        with pytest.raises(VectorIndexError, match="built with model"):

            class OtherEmbedder(FakeEmbedder):
                model_name = "different-model"

            VectorIndex(client, "coll", OtherEmbedder(dim=16))
    finally:
        client.close()


def test_dim_mismatch_raises() -> None:
    client = QdrantClient(":memory:")
    try:
        VectorIndex(client, "coll", FakeEmbedder(dim=16))
        with pytest.raises(VectorIndexError, match="dim"):
            VectorIndex(client, "coll", FakeEmbedder(dim=32))
    finally:
        client.close()


def test_batch_insert_scales(index: VectorIndex) -> None:
    bulk = [_chunk("synthetic.txt", i * 10, f"payload {i}") for i in range(200)]
    index.add(bulk)
    assert index.count() == 200
    sample = bulk[123]
    hits = index.search(sample.text, k=3)
    assert hits and hits[0][0] == sample.id
