"""Document chunking.

Three chunkers plus a dispatcher. Every chunk carries a stable, deterministic ID
(hash of source_path + start_offset + text) so reindexing can diff cleanly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import tiktoken

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_ENCODING = "cl100k_base"

TEXT_EXTENSIONS = {".txt", ".rst", ".log"}
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}
CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp"}

_FENCED_CODE_RE = re.compile(r"^([ \t]{0,3})(```+|~~~+)", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S.*$", re.MULTILINE)
_TOP_LEVEL_DEF_RE = re.compile(
    r"^(?:class|def|async\s+def|function|func|fn|public\s+class|interface|export\s+(?:async\s+)?function)\b",
    re.MULTILINE,
)

_encoder: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(DEFAULT_ENCODING)
    return _encoder


def token_count(text: str) -> int:
    return len(_enc().encode(text))


@dataclass(slots=True, frozen=True)
class Chunk:
    id: str
    source_path: str
    text: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


def make_chunk_id(source_path: str, start_offset: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(start_offset).encode("ascii"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _build_chunk(
    source_path: str,
    text: str,
    start_offset: int,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    end_offset = start_offset + len(text)
    return Chunk(
        id=make_chunk_id(source_path, start_offset, text),
        source_path=source_path,
        text=text,
        start_offset=start_offset,
        end_offset=end_offset,
        metadata=dict(metadata or {}),
    )


class TextChunker:
    """Token-windowed chunker using tiktoken cl100k_base."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        text: str,
        source_path: str,
        *,
        base_offset: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        enc = _enc()
        tokens = enc.encode(text)
        if not tokens:
            return []

        # Build a token-index -> char-offset map in O(text_size). The previous
        # implementation decoded `tokens[:i]` every iteration, which is
        # quadratic for long files.
        token_char_offsets = _token_char_offsets(enc, tokens, text)

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        n_tokens = len(tokens)
        i = 0
        while i < n_tokens:
            end_tok = min(i + self.chunk_size, n_tokens)
            char_start = token_char_offsets[i]
            char_end = token_char_offsets[end_tok]
            piece = text[char_start:char_end]
            if not piece:
                break
            # Safety net: if the byte map disagrees with what tiktoken decode
            # produces (rare BPE drift), fall back to nearby search so offsets
            # always satisfy `text[start:start+len(piece)] == piece`.
            if text[char_start : char_start + len(piece)] != piece:
                decoded = enc.decode(tokens[i:end_tok])
                if decoded:
                    piece = decoded
                    char_start = _find_near(text, piece, char_start)
            chunks.append(
                _build_chunk(source_path, piece, base_offset + char_start, metadata)
            )
            if end_tok >= n_tokens:
                break
            i += step
        return chunks


class MarkdownChunker:
    """Heading-aware Markdown chunker. Fenced code blocks stay atomic."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_fallback = TextChunker(chunk_size, chunk_overlap)

    def split(
        self,
        text: str,
        source_path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        fenced_spans = _fenced_code_spans(text)
        heading_positions = [
            m.start() for m in _HEADING_RE.finditer(text) if not _in_spans(m.start(), fenced_spans)
        ]
        if not heading_positions or heading_positions[0] != 0:
            heading_positions = [0, *heading_positions]
        heading_positions.append(len(text))

        chunks: list[Chunk] = []
        for idx in range(len(heading_positions) - 1):
            start = heading_positions[idx]
            end = heading_positions[idx + 1]
            section = text[start:end]
            section_stripped = section.strip("\n")
            if not section_stripped.strip():
                continue
            section_meta = {**(metadata or {}), "heading": _first_heading(section)}
            if token_count(section) <= self.chunk_size:
                chunks.append(_build_chunk(source_path, section, start, section_meta))
                continue
            chunks.extend(
                self._split_large_section(
                    section,
                    source_path,
                    base_offset=start,
                    metadata=section_meta,
                    fenced_spans=[
                        (s - start, e - start) for (s, e) in fenced_spans if s >= start and e <= end
                    ],
                )
            )
        return chunks

    def _split_large_section(
        self,
        section: str,
        source_path: str,
        *,
        base_offset: int,
        metadata: dict[str, Any],
        fenced_spans: list[tuple[int, int]],
    ) -> list[Chunk]:
        # Break section into atomic blocks (fenced code stays whole; prose splits on blank lines).
        blocks = _atomic_blocks(section, fenced_spans)
        chunks: list[Chunk] = []
        buffer: list[tuple[int, int, str]] = []  # (abs_start, rel_start, text)
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            merged_start = buffer[0][0]
            merged_text = "".join(b[2] for b in buffer)
            if token_count(merged_text) <= self.chunk_size:
                chunks.append(_build_chunk(source_path, merged_text, merged_start, metadata))
            else:
                # Block itself too large → fall back to token chunker over just this text.
                sub = self._text_fallback.split(
                    merged_text,
                    source_path,
                    base_offset=merged_start,
                    metadata=metadata,
                )
                chunks.extend(sub)
            buffer = []
            buffer_tokens = 0

        for rel_start, rel_end, block in blocks:
            block_tokens = token_count(block)
            if buffer_tokens + block_tokens > self.chunk_size and buffer:
                flush()
            buffer.append((base_offset + rel_start, rel_start, block))
            buffer_tokens += block_tokens
        flush()
        return chunks


_TREE_SITTER_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".hh": "cpp",
}


class CodeChunker:
    """Code chunker that prefers tree-sitter boundaries, with regex fallback.

    When the ``tree-sitter-language-pack`` extra is installed we walk the
    AST and emit one chunk per top-level definition plus any interleaved
    prose (imports, module docstrings, etc.). Without it we fall back to a
    regex scan for ``def`` / ``class`` / ``function`` / ``func`` / ``fn`` /
    ``interface`` and a token chunker for anything oversized.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        ext: str | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_fallback = TextChunker(chunk_size, chunk_overlap)
        self.ext = (ext or "").lower()

    def split(
        self,
        text: str,
        source_path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        ts_chunks = self._try_tree_sitter_split(text, source_path, metadata=metadata)
        if ts_chunks is not None:
            return ts_chunks
        return self._regex_split(text, source_path, metadata=metadata)

    # ---- tree-sitter path --------------------------------------------

    def _try_tree_sitter_split(
        self,
        text: str,
        source_path: str,
        *,
        metadata: dict[str, Any] | None,
    ) -> list[Chunk] | None:
        lang = _TREE_SITTER_LANG_BY_EXT.get(self.ext) if self.ext else None
        if lang is None:
            return None
        parser = _get_tree_sitter_parser(lang)
        if parser is None:
            return None
        try:
            src_bytes = text.encode("utf-8")
            tree = parser.parse(src_bytes)
        except Exception:  # noqa: BLE001 - parser bug or binding mismatch
            return None
        byte_to_char = _byte_to_char_map(text)
        chunks: list[Chunk] = []
        last_end = 0
        for node in tree.root_node.children:
            if node.start_byte > last_end:
                chunks.extend(
                    self._emit_block_by_bytes(
                        text,
                        byte_to_char,
                        last_end,
                        node.start_byte,
                        source_path,
                        metadata,
                    )
                )
            chunks.extend(
                self._emit_block_by_bytes(
                    text,
                    byte_to_char,
                    node.start_byte,
                    node.end_byte,
                    source_path,
                    metadata,
                )
            )
            last_end = node.end_byte
        if last_end < len(src_bytes):
            chunks.extend(
                self._emit_block_by_bytes(
                    text,
                    byte_to_char,
                    last_end,
                    len(src_bytes),
                    source_path,
                    metadata,
                )
            )
        return chunks

    def _emit_block_by_bytes(
        self,
        text: str,
        byte_to_char: list[int],
        byte_start: int,
        byte_end: int,
        source_path: str,
        metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        char_start = byte_to_char[byte_start] if byte_start < len(byte_to_char) else len(text)
        char_end = byte_to_char[byte_end] if byte_end < len(byte_to_char) else len(text)
        block = text[char_start:char_end]
        if not block.strip():
            return []
        if token_count(block) <= self.chunk_size:
            return [_build_chunk(source_path, block, char_start, metadata)]
        return self._text_fallback.split(
            block,
            source_path,
            base_offset=char_start,
            metadata=metadata,
        )

    # ---- regex fallback ----------------------------------------------

    def _regex_split(
        self,
        text: str,
        source_path: str,
        *,
        metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        starts = [m.start() for m in _TOP_LEVEL_DEF_RE.finditer(text)]
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        starts.append(len(text))

        chunks: list[Chunk] = []
        for idx in range(len(starts) - 1):
            start = starts[idx]
            end = starts[idx + 1]
            block = text[start:end]
            if not block.strip():
                continue
            if token_count(block) <= self.chunk_size:
                chunks.append(_build_chunk(source_path, block, start, metadata))
                continue
            chunks.extend(
                self._text_fallback.split(
                    block,
                    source_path,
                    base_offset=start,
                    metadata=metadata,
                )
            )
        return chunks


def chunk_file(
    path: str | Path,
    *,
    text: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    p = Path(path)
    content = text if text is not None else p.read_text(encoding="utf-8")
    chunker = _pick_chunker(p.suffix.lower(), chunk_size, chunk_overlap)
    return chunker.split(content, str(p), metadata=metadata)


def _pick_chunker(
    ext: str,
    chunk_size: int,
    chunk_overlap: int,
) -> TextChunker | MarkdownChunker | CodeChunker:
    if ext in MARKDOWN_EXTENSIONS:
        return MarkdownChunker(chunk_size, chunk_overlap)
    if ext in CODE_EXTENSIONS:
        return CodeChunker(chunk_size, chunk_overlap, ext=ext)
    return TextChunker(chunk_size, chunk_overlap)


def _fenced_code_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < len(text):
        m = _FENCED_CODE_RE.search(text, pos)
        if not m:
            break
        fence = m.group(2)
        start_line = text.rfind("\n", 0, m.start()) + 1
        # Find closing fence on its own line.
        close_re = re.compile(rf"^[ \t]{{0,3}}{re.escape(fence)}\s*$", re.MULTILINE)
        close = close_re.search(text, m.end())
        if close is None:
            spans.append((start_line, len(text)))
            break
        end = close.end()
        if end < len(text) and text[end] == "\n":
            end += 1
        spans.append((start_line, end))
        pos = end
    return spans


def _byte_to_char_map(text: str) -> list[int]:
    """Return a byte-offset → char-offset map for ``text`` in UTF-8."""
    src_bytes = text.encode("utf-8")
    total_bytes = len(src_bytes)
    mapping = [0] * (total_bytes + 1)
    byte_pos = 0
    for char_pos, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        for _ in range(width):
            if byte_pos < total_bytes:
                mapping[byte_pos] = char_pos
                byte_pos += 1
    mapping[total_bytes] = len(text)
    return mapping


_TREE_SITTER_PARSERS: dict[str, object] = {}
_TREE_SITTER_UNAVAILABLE: set[str] = set()


def _get_tree_sitter_parser(language: str):
    """Lazy-load a tree-sitter parser, caching successes and failures.

    Returns ``None`` if the ``tree_sitter_language_pack`` package is missing
    or the given language is not packaged for the current platform.
    """
    if language in _TREE_SITTER_PARSERS:
        return _TREE_SITTER_PARSERS[language]
    if language in _TREE_SITTER_UNAVAILABLE:
        return None
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)
    except Exception:  # noqa: BLE001 - optional dep, keep fallback path clean
        _TREE_SITTER_UNAVAILABLE.add(language)
        return None
    _TREE_SITTER_PARSERS[language] = parser
    return parser


def _token_char_offsets(enc, tokens: list[int], text: str) -> list[int]:
    """Return a list of length ``len(tokens) + 1`` mapping token boundaries
    to character offsets in ``text``.

    Works by (1) computing each token's byte width via
    ``enc.decode_single_token_bytes``, (2) building a byte-to-char map for
    the source text's UTF-8 encoding, and (3) indexing the byte offset at
    each token boundary through that map. Total work is O(len(text)) plus
    O(len(tokens)) — linear, so large files no longer trigger the old
    quadratic prefix-decode.
    """
    # Per-token byte widths.
    byte_widths: list[int] = [
        len(enc.decode_single_token_bytes(tok)) for tok in tokens
    ]
    cum_bytes: list[int] = [0] * (len(tokens) + 1)
    for i, width in enumerate(byte_widths):
        cum_bytes[i + 1] = cum_bytes[i] + width

    # byte_to_char[b] = index of the character that starts at byte offset b.
    # The array has len(utf8_bytes) + 1 entries so we can always look up the
    # tail position safely.
    byte_to_char = _byte_to_char_map(text)
    total_bytes = len(byte_to_char) - 1

    offsets: list[int] = []
    for byte_off in cum_bytes:
        if byte_off <= total_bytes:
            offsets.append(byte_to_char[byte_off])
        else:
            # BPE sometimes emits a trailing byte not present in the source
            # encoding; pin to end-of-text so callers still get a valid slice.
            offsets.append(len(text))
    return offsets


def _find_near(text: str, piece: str, expected: int) -> int:
    if not piece:
        return expected
    for delta in (0, -1, 1, -2, 2, -3, 3):
        cand = expected + delta
        if 0 <= cand <= len(text) - len(piece) and text[cand : cand + len(piece)] == piece:
            return cand
    lo = max(0, expected - 16)
    hi = min(len(text), expected + 16 + len(piece))
    idx = text.find(piece, lo, hi)
    if idx >= 0:
        return idx
    idx = text.find(piece)
    return idx if idx >= 0 else expected


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= pos < e:
            return True
    return False


def _first_heading(section: str) -> str:
    m = _HEADING_RE.search(section)
    if not m:
        return ""
    return m.group(0).lstrip("# ").strip()


def _atomic_blocks(
    section: str,
    fenced_spans: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """Split a Markdown section into atomic blocks.

    Fenced code spans become one block each. Non-fenced regions split on blank lines.
    Returns a list of (rel_start, rel_end, text) tuples covering the full section.
    """

    blocks: list[tuple[int, int, str]] = []
    cursor = 0
    for span_start, span_end in fenced_spans:
        if span_start > cursor:
            blocks.extend(_blank_line_blocks(section, cursor, span_start))
        blocks.append((span_start, span_end, section[span_start:span_end]))
        cursor = span_end
    if cursor < len(section):
        blocks.extend(_blank_line_blocks(section, cursor, len(section)))
    return blocks


def _blank_line_blocks(
    section: str,
    start: int,
    end: int,
) -> Iterable[tuple[int, int, str]]:
    region = section[start:end]
    if not region.strip():
        if region:
            yield (start, end, region)
        return
    local_cursor = 0
    for match in re.finditer(r"\n\s*\n", region):
        block_end = match.end()
        chunk_text = region[local_cursor:block_end]
        if chunk_text:
            yield (start + local_cursor, start + block_end, chunk_text)
        local_cursor = block_end
    if local_cursor < len(region):
        yield (start + local_cursor, end, region[local_cursor:])
