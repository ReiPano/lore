"""Tests for the FastEmbed wrapper.

The integration test downloads the real ``bge-small-en-v1.5`` model on first
run and is gated behind ``HYBRID_RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import importlib.util

import pytest

from hybrid_search.embeddings import DEFAULT_MODEL, Embedder


def test_embedder_defers_model_load() -> None:
    e = Embedder()
    assert e.model_name == DEFAULT_MODEL
    # `.embed([])` returns an empty list without ever instantiating the model.
    assert e.embed([]) == []
    assert e._model is None


def test_embedder_rejects_bad_batch() -> None:
    with pytest.raises(ValueError):
        Embedder(batch_size=0)


@pytest.mark.integration
def test_real_embed_has_expected_dim() -> None:
    if importlib.util.find_spec("fastembed") is None:
        pytest.skip("fastembed not installed")
    e = Embedder()
    vec = e.embed_one("hybrid search combines BM25 and dense vectors")
    assert vec.ndim == 1
    assert len(vec) == 384
    assert e.dim == 384


@pytest.mark.integration
def test_paraphrase_is_closer_than_random() -> None:
    if importlib.util.find_spec("fastembed") is None:
        pytest.skip("fastembed not installed")
    import numpy as np

    e = Embedder()
    anchor = e.embed_one("Reciprocal rank fusion merges two ranked lists")
    paraphrase = e.embed_one("RRF combines multiple ranking lists into one")
    unrelated = e.embed_one("The lifecycle of a monarch butterfly")

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cos(anchor, paraphrase) > cos(anchor, unrelated)
