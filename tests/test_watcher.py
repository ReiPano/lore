"""Tests for the incremental file watcher.

Tests drive the watcher through its public event hooks rather than spinning
up a real filesystem observer — that keeps them deterministic across macOS
FSEvents, Linux inotify, and CI sandboxes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hybrid_search.indexer import Indexer
from hybrid_search.lexical import LexicalIndex
from hybrid_search.vector import VectorIndex
from hybrid_search.watcher import IndexWatcher

from .conftest import FakeEmbedder


@pytest.fixture
def stack(tmp_path: Path):
    lex = LexicalIndex(tmp_path / "lex.sqlite3")
    vec = VectorIndex.in_memory("watcher_test", FakeEmbedder(dim=32))
    indexer = Indexer(lex, vec, chunk_size=500, chunk_overlap=50)
    yield lex, vec, indexer, tmp_path
    lex.close()
    vec.close()


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_upsert_then_flush_indexes_file(stack) -> None:
    lex, _, indexer, tmp = stack
    target = _write(tmp, "doc.md", "# Title\n\nContent.")
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0)
    watcher.on_upsert(target)
    assert watcher.has_pending()
    stats = watcher.flush()
    assert stats.chunks_added > 0
    assert lex.count() == stats.chunks_added
    assert not watcher.has_pending()


def test_delete_clears_chunks(stack) -> None:
    lex, _, indexer, tmp = stack
    target = _write(tmp, "doc.md", "# Title\n\nContent.")
    indexer.index_file(target)
    assert lex.count() > 0

    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0)
    watcher.on_delete(target)
    watcher.flush()
    assert lex.count() == 0


def test_rapid_updates_dedup(stack) -> None:
    lex, _, indexer, tmp = stack
    target = _write(tmp, "doc.md", "# Title\n\nFirst version of the content.")
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0)

    # Simulate editor save storm: many upsert events for the same path.
    for _ in range(10):
        watcher.on_upsert(target)

    stats = watcher.flush()
    # Only one upsert should have been executed, producing a finite number of chunks.
    assert stats.files_indexed == 1
    assert lex.count() == stats.chunks_added


def test_delete_then_upsert_results_in_upsert(stack) -> None:
    lex, _, indexer, tmp = stack
    target = _write(tmp, "doc.md", "# Title\n\nContent.")
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0)

    watcher.on_delete(target)
    watcher.on_upsert(target)
    watcher.flush()
    assert lex.count() > 0


def test_upsert_then_delete_results_in_delete(stack) -> None:
    lex, _, indexer, tmp = stack
    target = _write(tmp, "doc.md", "# Title\n\nContent.")
    indexer.index_file(target)
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0)

    watcher.on_upsert(target)
    watcher.on_delete(target)
    watcher.flush()
    assert lex.count() == 0


def test_pruned_directories_are_ignored(stack) -> None:
    lex, _, indexer, tmp = stack
    target = _write(tmp, ".venv/ignored.py", "def noop():\n    return 0\n")
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0)
    watcher.on_upsert(target)
    assert not watcher.has_pending()
    watcher.flush()
    assert lex.count() == 0


def test_gitignore_respected(stack) -> None:
    lex, _, indexer, tmp = stack
    _write(tmp, ".gitignore", "secrets.md\n")
    target = _write(tmp, "secrets.md", "# hush\n\ntop secret content")
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0, use_gitignore=True)
    watcher.on_upsert(target)
    assert not watcher.has_pending()
    watcher.flush()
    assert lex.count() == 0


def test_gitignore_opt_out(stack) -> None:
    lex, _, indexer, tmp = stack
    _write(tmp, ".gitignore", "secrets.md\n")
    target = _write(tmp, "secrets.md", "# hush\n\ntop secret content")
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0, use_gitignore=False)
    watcher.on_upsert(target)
    watcher.flush()
    assert lex.count() > 0


def test_debounce_worker_flushes_asynchronously(stack) -> None:
    lex, _, indexer, tmp = stack
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0.05)
    done = threading.Event()

    class Listener:
        def __init__(self, inner: Indexer) -> None:
            self.inner = inner

        def index_file(self, path):
            stats = self.inner.index_file(path)
            done.set()
            return stats

        def remove_path(self, path):
            return self.inner.remove_path(path)

    watcher.indexer = Listener(indexer)  # type: ignore[assignment]
    watcher._worker = threading.Thread(target=watcher._run, daemon=True)
    watcher._worker.start()
    try:
        target = _write(tmp, "doc.md", "# async\n\nflushed through worker")
        watcher.on_upsert(target)
        assert done.wait(timeout=2.0), "worker did not flush upsert in time"
        assert lex.count() > 0
    finally:
        watcher._stopped = True
        watcher._trigger.set()
        watcher._worker.join(timeout=2.0)


def test_start_stop_cycle_is_safe(stack) -> None:
    _, _, indexer, tmp = stack
    watcher = IndexWatcher(indexer, [tmp], debounce_seconds=0.01)
    watcher.start()
    time.sleep(0.05)
    watcher.stop()
    watcher.stop()  # idempotent
