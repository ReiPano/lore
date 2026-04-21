"""Tests for the indexing pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_search.indexer import Indexer
from hybrid_search.lexical import LexicalIndex
from hybrid_search.vector import VectorIndex

from .conftest import FakeEmbedder


@pytest.fixture
def stack(tmp_path: Path):
    lex = LexicalIndex(tmp_path / "lex.sqlite3")
    vec = VectorIndex.in_memory("indexer_test", FakeEmbedder(dim=32))
    yield lex, vec
    lex.close()
    vec.close()


def _write(tmp: Path, rel: str, body: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_index_single_file(tmp_path: Path, stack) -> None:
    lex, vec = stack
    target = _write(tmp_path, "doc.md", "# Hello\n\nSome content about search.\n")
    indexer = Indexer(lex, vec, chunk_size=500, chunk_overlap=50)
    stats = indexer.index_path(target)
    assert stats.files_indexed == 1
    assert stats.chunks_added > 0
    assert lex.count() == stats.chunks_added
    assert vec.count() == stats.chunks_added


def test_index_directory_walks_recursively(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "a.md", "# Alpha\n\nBody.\n")
    _write(tmp_path, "nested/b.py", "def beta():\n    return 1\n")
    _write(tmp_path, "nested/.venv/ignored.py", "def noop():\n    return 0\n")
    _write(tmp_path, "binary.bin", "binary-like content")

    indexer = Indexer(lex, vec)
    stats = indexer.index_path(tmp_path)
    assert stats.files_seen >= 3
    assert stats.files_indexed == 2  # md and py
    assert stats.files_skipped_unsupported >= 1  # .bin
    # Pruned dirs are skipped entirely.
    assert all(".venv" not in src for src in lex.sources())


def test_reindex_unchanged_is_noop(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "doc.md", "# Title\n\nStable content." * 20)
    indexer = Indexer(lex, vec)
    first = indexer.index_path(tmp_path)
    assert first.chunks_added > 0

    second = indexer.index_path(tmp_path)
    assert second.chunks_added == 0
    assert second.chunks_removed == 0
    assert second.files_skipped_unchanged == first.files_indexed


def test_modified_file_updates_chunks(tmp_path: Path, stack) -> None:
    lex, vec = stack
    target = _write(tmp_path, "doc.md", "# Title\n\nOriginal text about apples.")
    indexer = Indexer(lex, vec)
    indexer.index_path(target)
    original_chunks = lex.chunk_ids_for_source(str(target.resolve()))
    assert original_chunks

    target.write_text("# Title\n\nReplaced text about bananas.", encoding="utf-8")
    stats = indexer.index_path(target)
    assert stats.chunks_added > 0
    assert stats.chunks_removed > 0
    updated_chunks = lex.chunk_ids_for_source(str(target.resolve()))
    assert set(updated_chunks).isdisjoint(original_chunks)


def test_remove_path_file(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "keep.md", "# Keep\n\nThis stays.")
    gone = _write(tmp_path, "gone.md", "# Gone\n\nThis will disappear.")
    indexer = Indexer(lex, vec)
    indexer.index_path(tmp_path)

    stats = indexer.remove_path(gone)
    assert stats.files_removed == 1
    assert stats.chunks_removed > 0
    remaining_sources = lex.sources()
    assert all("gone.md" not in s for s in remaining_sources)


def test_remove_path_directory(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "keep/a.md", "# Keep\n\nHello.")
    _write(tmp_path, "drop/a.md", "# Drop\n\nWorld.")
    _write(tmp_path, "drop/b.py", "def f():\n    return 2\n")
    indexer = Indexer(lex, vec)
    indexer.index_path(tmp_path)

    stats = indexer.remove_path(tmp_path / "drop")
    assert stats.files_removed == 2
    assert all("drop" not in s for s in lex.sources())


def test_too_large_file_skipped(tmp_path: Path, stack) -> None:
    lex, vec = stack
    target = _write(tmp_path, "huge.md", "word " * 200)
    indexer = Indexer(lex, vec, max_file_bytes=10)
    stats = indexer.index_path(target)
    assert stats.files_skipped_too_large == 1
    assert stats.chunks_added == 0


def test_unsupported_extension_skipped(tmp_path: Path, stack) -> None:
    lex, vec = stack
    target = _write(tmp_path, "image.png", "fake png content")
    indexer = Indexer(lex, vec)
    stats = indexer.index_path(target)
    assert stats.files_skipped_unsupported == 1
    assert lex.count() == 0


def test_pdf_and_docx_skipped_when_parsers_missing(tmp_path: Path, stack, monkeypatch) -> None:
    lex, vec = stack
    pdf = _write(tmp_path, "doc.pdf", "not-a-real-pdf")
    docx = _write(tmp_path, "doc.docx", "not-a-real-docx")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"pypdf", "docx"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    indexer = Indexer(lex, vec)
    stats = indexer.index_path(pdf)
    assert stats.files_skipped_unsupported == 1
    stats = indexer.index_path(docx)
    assert stats.files_skipped_unsupported == 1


def test_vector_fail_fast_stops_after_threshold(tmp_path: Path, stack) -> None:
    lex, vec = stack
    # Write more files than the fail-fast limit so we can verify only the
    # threshold worth of errors get recorded before the run bails out.
    from hybrid_search.indexer import VECTOR_FAIL_FAST_LIMIT

    for i in range(VECTOR_FAIL_FAST_LIMIT + 3):
        _write(tmp_path, f"doc-{i}.md", f"# {i}\n\nBody of document {i}.")

    class BoomVector:
        def __init__(self, real):
            self.real = real

        def __getattr__(self, name):
            return getattr(self.real, name)

        def add(self, _):
            raise RuntimeError("qdrant 500")

    indexer = Indexer(lex, BoomVector(vec))
    stats = indexer.index_path(tmp_path)
    assert len(stats.errors) <= VECTOR_FAIL_FAST_LIMIT + 1
    assert any("aborting run" in msg for _, msg in stats.errors)
    assert stats.chunks_added == 0


def test_missing_path_reports_error(stack) -> None:
    lex, vec = stack
    indexer = Indexer(lex, vec)
    stats = indexer.index_path("/nonexistent/definitely/not/here")
    assert stats.errors
    assert "does not exist" in stats.errors[0][1]


def test_custom_exclude_dirs_prunes_tree(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "a.md", "# keep\n\nindexed")
    _write(tmp_path, "target/b.md", "# drop\n\nbuild artifact")
    _write(tmp_path, "nested/target/c.md", "# drop\n\nnested build")
    indexer = Indexer(lex, vec, exclude_dirs=["target"])
    stats = indexer.index_path(tmp_path)
    assert stats.files_indexed == 1
    assert all("target" not in s for s in lex.sources())


def test_exclude_patterns_skip_matching_files(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "app.js", "// keep real source")
    _write(tmp_path, "app.min.js", "// drop minified")
    _write(tmp_path, "package-lock.json", '{"drop":"me"}')  # .json not whitelisted anyway
    indexer = Indexer(
        lex,
        vec,
        exclude_patterns=["*.min.js", "package-lock.json"],
    )
    stats = indexer.index_path(tmp_path)
    assert stats.files_indexed == 1
    sources = lex.sources()
    assert any("app.js" in s and "min" not in s for s in sources)
    assert all("min.js" not in s for s in sources)


def test_sensitive_file_skip_by_name(tmp_path: Path, stack) -> None:
    lex, vec = stack
    real = _write(tmp_path, "note.md", "# Keep\n\nRegular content, safe.")
    secret = _write(tmp_path, ".env", "API_KEY=abcd1234\nDEBUG=true\n")
    # .env is in supported_extensions only if we add it; but it's also in
    # exclude_patterns, so it should be skipped regardless.
    indexer = Indexer(
        lex,
        vec,
        supported_extensions=[".md", ".env"],
        exclude_patterns=[".env", ".env.*"],
    )
    stats = indexer.index_path(tmp_path)
    sources = lex.sources()
    assert any("note.md" in s for s in sources)
    assert all(not s.endswith("/.env") for s in sources)
    assert stats.chunks_added > 0
    _ = real, secret  # silence unused


def test_sensitive_file_skip_by_content(tmp_path: Path, stack) -> None:
    lex, vec = stack
    _write(tmp_path, "keep.md", "# Keep\n\nNo secrets here.")
    _write(
        tmp_path,
        "leaky.md",
        "# Notes\n\nDon't commit this: sk-ant-abcdefghijklmnop012345.\n",
    )
    indexer = Indexer(
        lex,
        vec,
        exclude_content_patterns=[r"sk-ant-[A-Za-z0-9\-_]{20,}"],
    )
    stats = indexer.index_path(tmp_path)
    assert stats.files_skipped_sensitive == 1
    assert all("leaky.md" not in s for s in lex.sources())
    assert any("keep.md" in s for s in lex.sources())


def test_exclude_dirs_merged_with_builtin(tmp_path: Path, stack) -> None:
    lex, vec = stack
    # Built-in .git pruning + user-added .next.
    _write(tmp_path, ".git/hook.md", "# dropped by builtin")
    _write(tmp_path, ".next/page.md", "# dropped by user")
    _write(tmp_path, "keep.md", "# stays")
    indexer = Indexer(lex, vec, exclude_dirs=[".next"])
    stats = indexer.index_path(tmp_path)
    assert stats.files_indexed == 1
    srcs = lex.sources()
    assert all(".git" not in s and ".next" not in s for s in srcs)
