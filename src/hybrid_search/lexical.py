"""SQLite FTS5 BM25 lexical index.

Schema:
    chunks          regular table, source of truth for chunk metadata and text.
    chunks_fts      FTS5 virtual table, feeds BM25 ranking.

Both tables are written in a single transaction on `add`; FTS rows for updated
chunk IDs are deleted before insert to avoid duplicate postings.

FTS5 ``bm25()`` returns non-positive scores (lower = better). ``search`` negates
them so callers always see "higher is better".
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Iterable, Iterator

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    text          TEXT NOT NULL,
    start_offset  INTEGER NOT NULL,
    end_offset    INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    source_path UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


class LexicalIndex:
    """BM25 lexical index backed by SQLite FTS5."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        # `check_same_thread=False` + an explicit lock lets the search pipeline
        # call into the index from worker threads without opening a new
        # connection per request.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "LexicalIndex":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- writes ---------------------------------------------------------

    def add(self, chunks: Iterable[Chunk]) -> int:
        items = list(chunks)
        if not items:
            return 0
        now = time.time()
        meta_rows = [
            (
                c.id,
                c.source_path,
                c.text,
                c.start_offset,
                c.end_offset,
                json.dumps(c.metadata or {}),
                now,
            )
            for c in items
        ]
        fts_rows = [(c.id, c.text, c.source_path) for c in items]
        id_rows = [(c.id,) for c in items]
        with self._lock, self._transaction():
            self._conn.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, source_path, text, start_offset, end_offset, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    text = excluded.text,
                    start_offset = excluded.start_offset,
                    end_offset = excluded.end_offset,
                    metadata_json = excluded.metadata_json
                """,
                meta_rows,
            )
            self._conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", id_rows)
            self._conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, text, source_path) VALUES (?, ?, ?)",
                fts_rows,
            )
        return len(items)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        deleted = 0
        with self._lock, self._transaction():
            for batch in _chunks(ids, 500):
                placeholders = ",".join("?" for _ in batch)
                cur = self._conn.execute(
                    f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", batch
                )
                deleted += cur.rowcount or 0
                self._conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", batch
                )
        return deleted

    def delete_by_source(self, source_path: str) -> int:
        with self._lock, self._transaction():
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE source_path = ?", (source_path,)
            )
            deleted = cur.rowcount or 0
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE source_path = ?", (source_path,)
            )
        return deleted

    # ---- reads ----------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        fts_query = _sanitize_query(query)
        if not fts_query or k <= 0:
            return []
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, k),
            )
            rows = cur.fetchall()
        return [(cid, -float(score)) for cid, score in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT chunk_id, source_path, text, start_offset, end_offset, metadata_json
                FROM chunks WHERE chunk_id = ?
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
        return _row_to_chunk(row) if row else None

    def get_many(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return []
        by_id: dict[str, Chunk] = {}
        with self._lock:
            for batch in _chunks(ids, 500):
                placeholders = ",".join("?" for _ in batch)
                cur = self._conn.execute(
                    f"""
                    SELECT chunk_id, source_path, text, start_offset, end_offset, metadata_json
                    FROM chunks WHERE chunk_id IN ({placeholders})
                    """,
                    batch,
                )
                for row in cur.fetchall():
                    chunk = _row_to_chunk(row)
                    by_id[chunk.id] = chunk
        return [by_id[i] for i in ids if i in by_id]

    def chunk_ids_for_source(self, source_path: str) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE source_path = ? ORDER BY start_offset",
                (source_path,),
            )
            return [row[0] for row in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def sources(self) -> list[str]:
        with self._lock:
            return [
                row[0]
                for row in self._conn.execute(
                    "SELECT DISTINCT source_path FROM chunks ORDER BY source_path"
                )
            ]

    # ---- internals ------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise


def _sanitize_query(query: str) -> str:
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return ""
    parts: list[str] = []
    for tok in tokens:
        cleaned = tok.replace('"', '""')
        parts.append(f'"{cleaned}"')
    return " ".join(parts)


def _row_to_chunk(row: tuple) -> Chunk:
    return Chunk(
        id=row[0],
        source_path=row[1],
        text=row[2],
        start_offset=int(row[3]),
        end_offset=int(row[4]),
        metadata=json.loads(row[5]) if row[5] else {},
    )


def _chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
