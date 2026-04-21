"""End-to-end indexing pipeline.

Walks a path, reads each supported file, chunks it, then diffs the new chunk
IDs against what SQLite already stores for that source. Only the delta is
written to the two indexes — unchanged files are no-ops, so repeated calls are
cheap.

PDF and DOCX readers are optional (``pip install '.[parsers]'``). If the
parser package is missing, the file is skipped with a warning rather than
crashing the whole run.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .chunking import (
    CODE_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    TEXT_EXTENSIONS,
    chunk_file,
)
from .lexical import LexicalIndex
from .vector import VectorIndex

log = logging.getLogger(__name__)

DEFAULT_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS | CODE_EXTENSIONS | {".pdf", ".docx"}
)

PRUNED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".data",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_skipped_unsupported: int = 0
    files_skipped_too_large: int = 0
    files_skipped_sensitive: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def merge(self, other: "IndexStats") -> None:
        self.files_seen += other.files_seen
        self.files_indexed += other.files_indexed
        self.files_skipped_unchanged += other.files_skipped_unchanged
        self.files_skipped_unsupported += other.files_skipped_unsupported
        self.files_skipped_too_large += other.files_skipped_too_large
        self.files_skipped_sensitive += other.files_skipped_sensitive
        self.files_removed += other.files_removed
        self.chunks_added += other.chunks_added
        self.chunks_removed += other.chunks_removed
        self.errors.extend(other.errors)


SENSITIVE_SCAN_BYTES = 32 * 1024  # enough to catch most dotfile secrets


class Indexer:
    def __init__(
        self,
        lexical: LexicalIndex,
        vector: VectorIndex,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        supported_extensions: Iterable[str] | None = None,
        max_file_bytes: int = 10 * 1024 * 1024,
        show_progress: bool = False,
        exclude_dirs: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] | None = None,
        exclude_content_patterns: Iterable[str] | None = None,
    ) -> None:
        self.lexical = lexical
        self.vector = vector
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        exts = supported_extensions if supported_extensions else DEFAULT_SUPPORTED_EXTENSIONS
        self.supported_extensions = {e.lower() for e in exts}
        self.max_file_bytes = max_file_bytes
        self.show_progress = show_progress
        self.progress_label: str | None = None
        user_dirs = {d for d in (exclude_dirs or ()) if d}
        self.pruned_dirs: set[str] = PRUNED_DIRS | user_dirs
        self.exclude_patterns: list[str] = [p for p in (exclude_patterns or ()) if p]
        self.content_patterns: list[re.Pattern[str]] = _compile_content_patterns(
            exclude_content_patterns or ()
        )

    # ---- public API ---------------------------------------------------

    def index_path(self, path: str | Path) -> IndexStats:
        p = Path(path).resolve()
        stats = IndexStats()
        if not p.exists():
            stats.errors.append((str(p), "path does not exist"))
            return stats
        if p.is_file():
            stats.merge(self._index_one(p))
            return stats

        files = list(self._walk(p))
        stats.files_seen = len(files)
        for child_path in self._iter_with_progress(files):
            child_stats = self._index_one(child_path)
            # _index_one counts its own files_seen; drop that to avoid double counting.
            child_stats.files_seen = 0
            stats.merge(child_stats)
        return stats

    def index_file(self, path: str | Path) -> IndexStats:
        return self._index_one(Path(path).resolve())

    def remove_path(self, path: str | Path) -> IndexStats:
        p = Path(path).resolve()
        stats = IndexStats()
        targets: list[str] = []
        for src in self.lexical.sources():
            src_path = Path(src)
            if src_path == p:
                targets.append(src)
            elif p.is_dir():
                try:
                    if src_path.is_relative_to(p):
                        targets.append(src)
                except AttributeError:  # pragma: no cover - py<3.9 fallback
                    if str(src_path).startswith(str(p) + os.sep):
                        targets.append(src)
        for src in targets:
            ids = self.lexical.chunk_ids_for_source(src)
            if not ids:
                continue
            self.vector.delete(ids)
            self.lexical.delete(ids)
            stats.files_removed += 1
            stats.chunks_removed += len(ids)
        return stats

    # ---- internals ----------------------------------------------------

    def _walk(self, root: Path) -> Iterable[Path]:
        root_resolved = root.resolve()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in self.pruned_dirs and not d.startswith(".venv")
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if self._matches_exclude(path, root_resolved):
                    continue
                yield path

    def _matches_exclude(self, path: Path, root: Path) -> bool:
        if not self.exclude_patterns:
            return False
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            rel = Path(path.name)
        candidates = [path.name, str(rel), str(path)]
        for pattern in self.exclude_patterns:
            for cand in candidates:
                if fnmatch.fnmatch(cand, pattern):
                    return True
        return False

    def _iter_with_progress(self, files: list[Path]) -> Iterable[Path]:
        if not self.show_progress or not files:
            for f in files:
                log.info("indexing %s", f)
                yield f
            return
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )
        except ImportError:
            for f in files:
                log.info("indexing %s", f)
                yield f
            return
        label = self.progress_label or "Indexing"
        with Progress(
            TextColumn("[cyan]{task.description}[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[current]}"),
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=len(files), current="")
            for f in files:
                progress.update(task, current=str(f))
                log.info("indexing %s", f)
                yield f
                progress.advance(task)

    def _index_one(self, path: Path) -> IndexStats:
        stats = IndexStats(files_seen=1)
        ext = path.suffix.lower()
        if ext not in self.supported_extensions:
            stats.files_skipped_unsupported = 1
            return stats
        # pruned dir or pattern match → skip (covers single-file entry points).
        if set(path.parts) & self.pruned_dirs:
            stats.files_skipped_unsupported = 1
            return stats
        if self._matches_exclude(path, path.parent):
            stats.files_skipped_unsupported = 1
            return stats
        try:
            size = path.stat().st_size
        except OSError as exc:
            stats.errors.append((str(path), f"stat failed: {exc}"))
            return stats
        if size > self.max_file_bytes:
            stats.files_skipped_too_large = 1
            return stats
        try:
            text = _read_as_text(path)
        except Exception as exc:  # noqa: BLE001 - surface to caller, don't crash run
            stats.errors.append((str(path), f"read failed: {exc}"))
            return stats
        if text is None:
            stats.files_skipped_unsupported = 1
            return stats

        if self.content_patterns and _content_matches_sensitive(
            text, self.content_patterns
        ):
            log.info("skipping sensitive-content file %s", path)
            stats.files_skipped_sensitive = 1
            return stats

        new_chunks = chunk_file(
            path,
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        existing_ids = set(self.lexical.chunk_ids_for_source(str(path)))
        new_ids = {c.id for c in new_chunks}
        to_add = [c for c in new_chunks if c.id not in existing_ids]
        to_remove = list(existing_ids - new_ids)

        if to_remove:
            try:
                self.vector.delete(to_remove)
                self.lexical.delete(to_remove)
                stats.chunks_removed = len(to_remove)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append((str(path), f"delete failed: {exc}"))
                return stats

        if to_add:
            try:
                self.vector.add(to_add)
                self.lexical.add(to_add)
                stats.chunks_added = len(to_add)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append((str(path), f"add failed: {exc}"))
                # Best-effort rollback of whichever side landed first.
                try:
                    self.vector.delete([c.id for c in to_add])
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.lexical.delete([c.id for c in to_add])
                except Exception:  # noqa: BLE001
                    pass
                return stats

        if to_add or to_remove:
            stats.files_indexed = 1
        else:
            stats.files_skipped_unchanged = 1
        return stats


def _compile_content_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        if not raw:
            continue
        try:
            compiled.append(re.compile(raw))
        except re.error as exc:
            log.warning("ignoring invalid content pattern %r: %s", raw, exc)
    return compiled


def _content_matches_sensitive(
    text: str,
    patterns: list[re.Pattern[str]],
) -> bool:
    """Scan only the head of the file so we do not pay for every byte.

    Most secrets live in dotfiles or the first few lines of a script. Bounding
    to ``SENSITIVE_SCAN_BYTES`` keeps the scan cheap on large source files.
    """
    window = text[:SENSITIVE_SCAN_BYTES]
    return any(p.search(window) for p in patterns)


def _read_as_text(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:  # noqa: BLE001
            return None


def _read_pdf(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; skipping %s (install extras 'parsers')", path)
        return None
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("pdf read failed for %s: %s", path, exc)
        return None
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            parts.append("")
    return "\n\n".join(parts)


def _read_docx(path: Path) -> str | None:
    try:
        import docx  # type: ignore[import-not-found]
    except ImportError:
        log.warning(
            "python-docx not installed; skipping %s (install extras 'parsers')", path
        )
        return None
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("docx read failed for %s: %s", path, exc)
        return None
    return "\n\n".join(p.text for p in document.paragraphs)
