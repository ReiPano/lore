"""Smoke test: every module imports without error."""

import importlib

import pytest

MODULES = [
    "hybrid_search",
    "hybrid_search.config",
    "hybrid_search.chunking",
    "hybrid_search.embeddings",
    "hybrid_search.lexical",
    "hybrid_search.vector",
    "hybrid_search.fusion",
    "hybrid_search.rerank",
    "hybrid_search.indexer",
    "hybrid_search.search",
    "hybrid_search.watcher",
    "hybrid_search.api",
    "hybrid_search.mcp_server",
    "hybrid_search.cli",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_config_loads() -> None:
    from pathlib import Path

    from hybrid_search.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.collection_name == "hybrid_chunks"
    assert cfg.embedding_dim == 384
    assert cfg.chunk_size > 0
