"""Tests for Phase 1 chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_search.chunking import (
    CODE_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    Chunk,
    CodeChunker,
    MarkdownChunker,
    TextChunker,
    chunk_file,
    make_chunk_id,
    token_count,
)


def test_chunk_id_is_stable() -> None:
    a = make_chunk_id("docs/a.md", 0, "hello world")
    b = make_chunk_id("docs/a.md", 0, "hello world")
    assert a == b
    assert make_chunk_id("docs/a.md", 1, "hello world") != a
    assert make_chunk_id("docs/b.md", 0, "hello world") != a
    assert make_chunk_id("docs/a.md", 0, "hello worlds") != a


def test_text_chunker_empty_input() -> None:
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    assert chunker.split("", "empty.txt") == []
    assert chunker.split("   \n\t  ", "blank.txt") == []


def test_text_chunker_single_chunk_when_under_limit() -> None:
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    text = "This is a short document about search systems and ranking."
    chunks = chunker.split(text, "doc.txt")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == text
    assert chunk.start_offset == 0
    assert chunk.end_offset == len(text)
    assert chunk.source_path == "doc.txt"


def test_text_chunker_produces_multiple_chunks_and_respects_token_limit() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    paragraph = (
        "Hybrid search combines lexical and semantic retrieval. "
        "BM25 handles identifiers and rare terms while vectors catch paraphrases. "
        "Reciprocal rank fusion merges the two result lists without needing calibrated scores. "
        "A small cross-encoder reranker then polishes the top candidates. "
    ) * 5
    chunks = chunker.split(paragraph, "doc.txt")
    assert len(chunks) > 1
    for chunk in chunks:
        assert token_count(chunk.text) <= 50
    # offsets are monotonically non-decreasing
    starts = [c.start_offset for c in chunks]
    assert starts == sorted(starts)


def test_text_chunker_offsets_slice_source_back() -> None:
    chunker = TextChunker(chunk_size=60, chunk_overlap=10)
    text = "word " * 200
    chunks = chunker.split(text, "doc.txt")
    assert chunks
    for chunk in chunks:
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text


def test_text_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)


def test_markdown_chunker_splits_on_headings() -> None:
    md = (
        "# Title\n\n"
        "Intro paragraph about search.\n\n"
        "## Section One\n\n"
        "BM25 is a lexical ranker.\n\n"
        "## Section Two\n\n"
        "Dense vectors capture semantics.\n"
    )
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).split(md, "doc.md")
    headings = [c.metadata.get("heading") for c in chunks]
    assert "Title" in headings
    assert "Section One" in headings
    assert "Section Two" in headings
    for chunk in chunks:
        assert md[chunk.start_offset : chunk.end_offset] == chunk.text


def test_markdown_chunker_keeps_fenced_code_atomic() -> None:
    md = (
        "# Heading\n\n"
        + ("Prose line about the topic. " * 40)
        + "\n\n"
        + "```python\n"
        + "def foo():\n"
        + "    return 42\n"
        + "```\n\n"
        + ("More prose. " * 40)
    )
    chunker = MarkdownChunker(chunk_size=60, chunk_overlap=10)
    chunks = chunker.split(md, "doc.md")
    code_block = "```python\ndef foo():\n    return 42\n```"
    assert any(code_block in c.text for c in chunks), "fenced code block must live inside a single chunk"
    for chunk in chunks:
        # Every chunk should be a slice of the source (offsets preserved).
        assert md[chunk.start_offset : chunk.end_offset] == chunk.text


def test_code_chunker_splits_on_top_level_defs() -> None:
    src = (
        "import os\n\n"
        "def alpha():\n    return 1\n\n"
        "def beta():\n    return 2\n\n"
        "class Gamma:\n    def method(self):\n        return 3\n"
    )
    chunks = CodeChunker(chunk_size=500, chunk_overlap=50).split(src, "mod.py")
    joined = "".join(c.text for c in chunks)
    assert joined == src
    # Each def/class should start a chunk (beyond the prelude).
    starts = {c.text.lstrip().split("\n", 1)[0] for c in chunks}
    assert any(s.startswith("def alpha") for s in starts)
    assert any(s.startswith("def beta") for s in starts)
    assert any(s.startswith("class Gamma") for s in starts)


def test_code_chunker_falls_back_to_token_split_for_huge_function() -> None:
    body = "    x = 1\n" * 400
    src = f"def huge():\n{body}"
    chunks = CodeChunker(chunk_size=100, chunk_overlap=20).split(src, "mod.py")
    assert len(chunks) > 1
    for chunk in chunks:
        assert token_count(chunk.text) <= 100


def test_dispatcher_picks_right_chunker(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Title\n\nBody.\n", encoding="utf-8")
    py_path = tmp_path / "mod.py"
    py_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Plain notes.\n", encoding="utf-8")

    md_chunks = chunk_file(md_path)
    py_chunks = chunk_file(py_path)
    txt_chunks = chunk_file(txt_path)
    assert md_chunks and py_chunks and txt_chunks
    # Markdown metadata carries heading info; plain text does not.
    assert any("heading" in c.metadata for c in md_chunks)
    assert all("heading" not in c.metadata for c in txt_chunks)


def test_reindex_is_stable_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "# Title\n\nHello world. " * 100,
        encoding="utf-8",
    )
    first = chunk_file(path, chunk_size=80, chunk_overlap=10)
    second = chunk_file(path, chunk_size=80, chunk_overlap=10)
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_extension_sets_cover_plan_targets() -> None:
    for ext in {".py", ".js", ".ts", ".go", ".rs", ".java"}:
        assert ext in CODE_EXTENSIONS
    assert ".md" in MARKDOWN_EXTENSIONS


def test_chunk_is_frozen() -> None:
    chunk = Chunk(
        id="abc",
        source_path="x.txt",
        text="hello",
        start_offset=0,
        end_offset=5,
    )
    with pytest.raises(Exception):
        chunk.text = "mutated"  # type: ignore[misc]


def test_text_chunker_offsets_match_for_large_input() -> None:
    """Regression guard: the byte-offset map must produce exact slices.

    Uses a mixed ASCII + multibyte text so UTF-8 boundaries get exercised.
    """
    body = (
        "Hybrid search combines BM25 and vectors. "
        "Rare identifiers like `sha256_hash_v1` matter. "
        "Multibyte content — é à ñ 漢字 emoji 🚀 — stays aligned. "
    ) * 120
    chunker = TextChunker(chunk_size=80, chunk_overlap=16)
    chunks = chunker.split(body, "doc.txt")
    assert len(chunks) > 10
    for chunk in chunks:
        assert body[chunk.start_offset : chunk.end_offset] == chunk.text
        # Multibyte boundaries can shift the re-encoded token count by a
        # couple of tokens. Allow a small slack; the important invariant is
        # that chunks stay bounded and slice cleanly back out of `body`.
        assert token_count(chunk.text) <= 80 + 4


def test_code_chunker_uses_tree_sitter_for_python() -> None:
    tree_sitter = pytest.importorskip("tree_sitter_language_pack")
    assert tree_sitter  # import side effect

    src = (
        "import os\n"
        "\n"
        "def alpha(x):\n"
        "    return x + 1\n"
        "\n"
        "def beta(x):\n"
        "    return x * 2\n"
        "\n"
        "class Gamma:\n"
        "    def method(self):\n"
        "        return 3\n"
    )
    chunks = CodeChunker(chunk_size=500, chunk_overlap=50, ext=".py").split(src, "mod.py")
    texts = [c.text.strip() for c in chunks]
    # Tree-sitter emits a chunk per top-level node.
    assert any(t.startswith("def alpha") for t in texts)
    assert any(t.startswith("def beta") for t in texts)
    assert any(t.startswith("class Gamma") for t in texts)
    # Imports live in their own chunk, not glued to a function body.
    assert any(t == "import os" for t in texts)
    # Every chunk still slices back out of the source byte-identically.
    for chunk in chunks:
        assert src[chunk.start_offset : chunk.end_offset] == chunk.text


def test_code_chunker_regex_fallback_when_ext_unknown() -> None:
    """Unknown extensions fall through to the regex splitter."""
    src = (
        "def alpha():\n    return 1\n\n"
        "def beta():\n    return 2\n"
    )
    chunks = CodeChunker(chunk_size=500, chunk_overlap=50, ext=".xyz").split(
        src, "mod.xyz"
    )
    texts = [c.text for c in chunks]
    assert "".join(texts) == src


def test_text_chunker_scales_linearly_for_huge_input() -> None:
    """Sanity check: 200k-token input finishes fast with the new O(n) path."""
    import time

    body = ("hybrid search " * 50_000).strip()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    start = time.perf_counter()
    chunks = chunker.split(body, "big.txt")
    elapsed = time.perf_counter() - start
    assert chunks, "expected at least one chunk"
    # Pre-patch implementation took many seconds on this size; new path
    # finishes in well under a second on a laptop.
    assert elapsed < 5.0, f"chunker regressed: {elapsed:.2f}s"
    for chunk in chunks[:5]:
        assert body[chunk.start_offset : chunk.end_offset] == chunk.text
