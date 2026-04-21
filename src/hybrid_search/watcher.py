"""File watcher that feeds incremental updates into the indexer.

Subscribes to one or more paths via ``watchdog``. Creates and modifications
queue the affected file for upsert; deletes queue it for removal. Events are
debounced within a configurable window so an editor's rapid-save storm
collapses into a single reindex pass.

Gitignore support is opt-in and loads a ``.gitignore`` from each watched root
using ``pathspec``. Pruned directories (``.git``, ``node_modules``, etc.) are
always filtered out.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import PRUNED_DIRS, IndexStats, Indexer

log = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 2.0


class IndexWatcher:
    """Bridge between watchdog events and the :class:`Indexer`."""

    def __init__(
        self,
        indexer: Indexer,
        paths: Iterable[str | Path],
        *,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        use_gitignore: bool = True,
    ) -> None:
        self.indexer = indexer
        self.paths = [Path(p).resolve() for p in paths]
        self.debounce_seconds = max(0.0, float(debounce_seconds))
        self.use_gitignore = use_gitignore

        self._pending_upsert: set[Path] = set()
        self._pending_delete: set[Path] = set()
        self._state_lock = threading.Lock()
        self._trigger = threading.Event()
        self._worker: threading.Thread | None = None
        self._observer: Observer | None = None
        self._stopped = False
        self._gitignores: list[_GitIgnoreSpec] = (
            _load_gitignores(self.paths) if use_gitignore else []
        )

    # ---- public API ---------------------------------------------------

    def on_upsert(self, path: str | Path) -> None:
        p = Path(path).resolve()
        if not self._should_track(p):
            return
        with self._state_lock:
            self._pending_delete.discard(p)
            self._pending_upsert.add(p)
        self._trigger.set()

    def on_delete(self, path: str | Path) -> None:
        p = Path(path).resolve()
        with self._state_lock:
            self._pending_upsert.discard(p)
            self._pending_delete.add(p)
        self._trigger.set()

    def flush(self) -> IndexStats:
        """Drain pending events and apply them synchronously."""
        with self._state_lock:
            upserts = sorted(self._pending_upsert)
            deletes = sorted(self._pending_delete)
            self._pending_upsert.clear()
            self._pending_delete.clear()
        stats = IndexStats()
        for p in deletes:
            stats.merge(self.indexer.remove_path(p))
        for p in upserts:
            stats.merge(self.indexer.index_file(p))
        return stats

    def has_pending(self) -> bool:
        with self._state_lock:
            return bool(self._pending_upsert or self._pending_delete)

    def start(self) -> None:
        if self._observer is not None:
            return
        handler = _EventHandler(self)
        self._observer = Observer()
        for p in self.paths:
            if p.exists():
                self._observer.schedule(handler, str(p), recursive=True)
        self._observer.start()
        self._stopped = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stopped = True
        self._trigger.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

    def __enter__(self) -> "IndexWatcher":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ---- internals ----------------------------------------------------

    def _run(self) -> None:
        while not self._stopped:
            triggered = self._trigger.wait(timeout=self.debounce_seconds or None)
            if self._stopped:
                break
            if triggered:
                self._trigger.clear()
                # Sleep through the debounce window so rapid bursts collapse.
                if self.debounce_seconds > 0:
                    time.sleep(self.debounce_seconds)
            try:
                self.flush()
            except Exception:  # noqa: BLE001
                log.exception("watcher flush failed")

    def _should_track(self, path: Path) -> bool:
        pruned = getattr(self.indexer, "pruned_dirs", PRUNED_DIRS)
        if set(path.parts) & pruned:
            return False
        for spec in self._gitignores:
            if spec.matches(path):
                return False
        patterns = getattr(self.indexer, "exclude_patterns", ())
        if patterns:
            import fnmatch

            name = path.name
            full = str(path)
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full, pattern):
                    return False
        return True


class _EventHandler(FileSystemEventHandler):
    def __init__(self, watcher: IndexWatcher) -> None:
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.watcher.on_upsert(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.watcher.on_upsert(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self.watcher.on_delete(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self.watcher.on_delete(event.src_path)
        dest = getattr(event, "dest_path", None)
        if dest:
            self.watcher.on_upsert(dest)


@dataclass
class _GitIgnoreSpec:
    root: Path
    spec: object  # pathspec.PathSpec but imported lazily

    def matches(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False
        try:
            return bool(self.spec.match_file(str(rel)))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return False


def _load_gitignores(paths: list[Path]) -> list[_GitIgnoreSpec]:
    try:
        import pathspec
    except ImportError:
        return []
    specs: list[_GitIgnoreSpec] = []
    for root in paths:
        gi = root / ".gitignore"
        if not gi.is_file():
            continue
        lines = _read_gitignore(gi)
        compiled = _compile_gitignore(pathspec, lines)
        if compiled is None:
            continue
        specs.append(_GitIgnoreSpec(root=root, spec=compiled))
    return specs


def _read_gitignore(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return []


def _compile_gitignore(pathspec_module, lines: list[str]):
    # Prefer the non-deprecated pattern name where available.
    for style in ("gitignore", "gitwildmatch"):
        try:
            return pathspec_module.PathSpec.from_lines(style, lines)
        except (ValueError, LookupError):
            continue
        except Exception:  # noqa: BLE001
            return None
    return None
