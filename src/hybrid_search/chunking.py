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

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        i = 0
        while i < len(tokens):
            prefix = enc.decode(tokens[:i]) if i else ""
            window = tokens[i : i + self.chunk_size]
            piece = enc.decode(window)
            if not piece:
                break
            start_offset = len(prefix)
            if text[start_offset : start_offset + len(piece)] != piece:
                # Rare BPE boundary drift: search a tight window then fall back.
                start_offset = _find_near(text, piece, start_offset)
            chunks.append(_build_chunk(source_path, piece, base_offset + start_offset, metadata))
            if i + self.chunk_size >= len(tokens):
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


class CodeChunker:
    """Code chunker that splits on top-level definitions, with line-count fallback."""

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
        return CodeChunker(chunk_size, chunk_overlap)
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
