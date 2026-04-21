"""Tests for the cross-encoder reranker wrapper."""

from __future__ import annotations

from hybrid_search import rerank as rerank_module
from hybrid_search.rerank import DEFAULT_RERANKER_MODEL, FALLBACK_RERANKER_MODELS, Reranker


class _StubRerankModel:
    def __init__(self, model_name: str, **_) -> None:
        self.model_name = model_name

    def rerank(self, query: str, documents):
        # Trivial scoring: keyword count. Gets exercised only if a model loads.
        return [float(d.lower().count(query.lower())) for d in documents]


def test_reranker_falls_back_when_primary_unavailable(monkeypatch) -> None:
    attempts: list[str] = []

    class _StubEncoderModule:
        def __call__(self, model_name, cache_dir=None):
            attempts.append(model_name)
            if model_name == DEFAULT_RERANKER_MODEL:
                raise RuntimeError("not available in this environment")
            return _StubRerankModel(model_name)

    def fake_import(_):
        # Return an object with TextCrossEncoder attribute behaving like our stub.
        class Pkg:
            TextCrossEncoder = _StubEncoderModule()

        return Pkg()

    real_import = __import__

    def import_hook(name, *a, **kw):
        if name == "fastembed.rerank.cross_encoder":
            return fake_import(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", import_hook)

    r = Reranker()
    scores = r.score("alpha", ["alpha beta", "nothing"])
    assert scores == [1.0, 0.0]
    # First attempt was the primary; fallback took over.
    assert attempts[0] == DEFAULT_RERANKER_MODEL
    assert r.active_model_name in FALLBACK_RERANKER_MODELS


def test_reranker_empty_texts_short_circuits() -> None:
    r = Reranker()
    # Must not trigger any model load when there's nothing to score.
    assert r.score("anything", []) == []
    assert r._model is None


def test_default_model_constant_is_bge_v2_m3() -> None:
    assert DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert "BAAI/bge-reranker-base" in FALLBACK_RERANKER_MODELS
    assert rerank_module.FALLBACK_RERANKER_MODELS[-1].endswith("MiniLM-L-6-v2")
