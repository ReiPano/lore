"""Tests for the `hybrid-search projects add-all` command.

The CLI expects a real config + live Qdrant when it builds the indexer, so we
call the internal handler directly with a locally-wired stack instead of
invoking ``main()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hybrid_search.cli import _cmd_projects_add_all
from hybrid_search.config import Config
from hybrid_search.indexer import Indexer
from hybrid_search.lexical import LexicalIndex
from hybrid_search.projects import ProjectStore
from hybrid_search.vector import VectorIndex

from .conftest import FakeEmbedder


def _config(tmp_path: Path) -> Config:
    return Config(
        index_path=tmp_path / "lex.sqlite3",
        qdrant_url="unused",
        collection_name="add_all_test",
        embedding_model="fake-hash-v1",
        embedding_dim=32,
        rerank_model="",
        rerank_enabled=False,
        chunk_size=500,
        chunk_overlap=50,
        fusion_k=60,
        bm25_weight=1.0,
        vector_weight=1.0,
        retrieval_k_per_index=10,
        rerank_top_n=5,
        default_result_k=5,
        watch_paths=[],
        max_file_bytes=10 * 1024 * 1024,
        supported_extensions=[".md", ".txt"],
        exclude_dirs=[],
        exclude_patterns=[],
    )


@pytest.fixture
def wired(tmp_path: Path, monkeypatch):
    cfg = _config(tmp_path)
    lex = LexicalIndex(cfg.index_path)
    vec = VectorIndex.in_memory(cfg.collection_name, FakeEmbedder(dim=32))
    indexer = Indexer(
        lex,
        vec,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        supported_extensions=cfg.supported_extensions,
        max_file_bytes=cfg.max_file_bytes,
        show_progress=False,
    )

    # Reroute _build_stack / _build_indexer to reuse this one stack.
    import hybrid_search.cli as cli_mod

    # CLI commands close lex/vec at the end — but tests still need them open
    # to make assertions, so swap the closers for no-ops during the test.
    monkeypatch.setattr(type(lex), "close", lambda self: None)
    monkeypatch.setattr(type(vec), "close", lambda self: None)
    monkeypatch.setattr(cli_mod, "_build_stack", lambda _cfg: (lex, vec, None))
    monkeypatch.setattr(cli_mod, "_build_indexer", lambda _cfg, _l, _v: indexer)
    # watcher_running always returns None in tests so the "restart watcher" hint skips.
    monkeypatch.setattr(cli_mod, "watcher_running", lambda _dir: None)

    yield cfg, lex, vec
    # Monkeypatched close was a no-op; flush directly at teardown.
    try:
        lex._conn.close()
    except Exception:
        pass


def _make_project(root: Path, name: str, body: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "notes.md").write_text(body, encoding="utf-8")
    return d


def test_add_all_registers_each_subfolder(tmp_path: Path, wired) -> None:
    cfg, lex, _vec = wired
    parent = tmp_path / "workspace"
    parent.mkdir()
    _make_project(parent, "alpha", "# Alpha\n\nAlpha content about search.")
    _make_project(parent, "beta", "# Beta\n\nBeta content about vectors.")
    _make_project(parent, "gamma", "# Gamma\n\nGamma content about fusion.")
    # Non-dirs and hidden dirs should be ignored.
    (parent / "README.txt").write_text("skip me", encoding="utf-8")
    (parent / ".hidden").mkdir()
    (parent / "node_modules").mkdir()

    args = argparse.Namespace(path=str(parent), no_watch=False, skip_index=False)
    rc = _cmd_projects_add_all(args, cfg)
    assert rc == 0

    store = ProjectStore(tmp_path / "projects.json")
    names = {p.name for p in store.load()}
    assert names == {"alpha", "beta", "gamma"}
    # Indexed chunks landed in the shared stack.
    assert lex.count() > 0


def test_add_all_skip_index_does_not_touch_stack(tmp_path: Path, wired) -> None:
    cfg, lex, _vec = wired
    parent = tmp_path / "repos"
    parent.mkdir()
    _make_project(parent, "one", "# One\n\nBody.")

    args = argparse.Namespace(path=str(parent), no_watch=True, skip_index=True)
    rc = _cmd_projects_add_all(args, cfg)
    assert rc == 0
    assert lex.count() == 0

    store = ProjectStore(tmp_path / "projects.json")
    projects = store.load()
    assert len(projects) == 1
    assert projects[0].watch is False


def test_add_all_errors_when_path_not_directory(tmp_path: Path, wired) -> None:
    cfg, _lex, _vec = wired
    target = tmp_path / "file.txt"
    target.write_text("not a dir", encoding="utf-8")
    args = argparse.Namespace(path=str(target), no_watch=False, skip_index=False)
    rc = _cmd_projects_add_all(args, cfg)
    assert rc == 1


def test_add_all_empty_parent_is_noop(tmp_path: Path, wired) -> None:
    cfg, lex, _vec = wired
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    args = argparse.Namespace(path=str(empty), no_watch=False, skip_index=False)
    rc = _cmd_projects_add_all(args, cfg)
    assert rc == 0
    assert lex.count() == 0
