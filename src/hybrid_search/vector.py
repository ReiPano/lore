"""Qdrant vector index wrapper.

Stores one point per chunk. The chunk_id (16 hex chars) is mapped to a uint64
Qdrant point ID so deletes and upserts can address points directly without a
secondary lookup. Payload carries the chunk_id, source_path, a short text
preview, and a ``kind`` marker so we can filter out the reserved meta point.

Point id 0 is reserved: it holds a zero-vector payload recording which
embedding model built the collection. ``_ensure_collection`` refuses to open a
collection whose recorded model differs from the current embedder.
"""

from __future__ import annotations

import logging
from typing import Iterable, Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .chunking import Chunk

log = logging.getLogger(__name__)

META_POINT_ID = 0
PREVIEW_CHARS = 200


class VectorIndexError(RuntimeError):
    pass


class EmbedderProtocol(Protocol):
    model_name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]: ...

    def embed_one(self, text: str) -> np.ndarray: ...


class VectorIndex:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder: EmbedderProtocol,
        *,
        dim: int | None = None,
    ) -> None:
        self._client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.dim = dim if dim is not None else embedder.dim
        self._ensure_collection()

    @classmethod
    def connect(
        cls,
        url: str,
        collection_name: str,
        embedder: EmbedderProtocol,
        *,
        dim: int | None = None,
    ) -> "VectorIndex":
        return cls(QdrantClient(url=url), collection_name, embedder, dim=dim)

    @classmethod
    def in_memory(
        cls,
        collection_name: str,
        embedder: EmbedderProtocol,
        *,
        dim: int | None = None,
    ) -> "VectorIndex":
        return cls(QdrantClient(":memory:"), collection_name, embedder, dim=dim)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - close is best-effort
            pass

    # ---- collection setup ----------------------------------------------

    def _ensure_collection(self) -> None:
        exists = False
        try:
            exists = bool(self._client.collection_exists(self.collection_name))
        except Exception:  # noqa: BLE001 - treat any error as "does not exist yet"
            exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        else:
            existing_dim = self._collection_dim()
            if existing_dim != self.dim:
                raise VectorIndexError(
                    f"Collection {self.collection_name!r} has dim {existing_dim}, "
                    f"embedder produces dim {self.dim}"
                )
            meta = self._read_meta()
            recorded_model = (meta or {}).get("embedding_model")
            if recorded_model and recorded_model != self.embedder.model_name:
                raise VectorIndexError(
                    f"Collection {self.collection_name!r} was built with model "
                    f"{recorded_model!r}, current embedder is {self.embedder.model_name!r}"
                )
        self._write_meta()

    def _collection_dim(self) -> int:
        info = self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)
        # Named vector collections return a mapping; use the first entry.
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            return int(getattr(first, "size", 0))
        raise VectorIndexError("Unable to read collection dimension")

    def _read_meta(self) -> dict | None:
        try:
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[META_POINT_ID],
                with_payload=True,
            )
        except Exception:  # noqa: BLE001
            return None
        if not points:
            return None
        return points[0].payload or None

    def _write_meta(self) -> None:
        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=META_POINT_ID,
                    vector=[0.0] * self.dim,
                    payload={
                        "kind": "meta",
                        "embedding_model": self.embedder.model_name,
                    },
                )
            ],
            wait=True,
        )

    # ---- writes ---------------------------------------------------------

    def add(self, chunks: Iterable[Chunk]) -> int:
        items = list(chunks)
        if not items:
            return 0
        vectors = self.embedder.embed([c.text for c in items])
        if len(vectors) != len(items):
            raise VectorIndexError(
                f"embedder returned {len(vectors)} vectors for {len(items)} chunks"
            )
        points = [
            qmodels.PointStruct(
                id=_chunk_point_id(c.id),
                vector=_to_list(vectors[i]),
                payload={
                    "kind": "chunk",
                    "chunk_id": c.id,
                    "source_path": c.source_path,
                    "preview": c.text[:PREVIEW_CHARS],
                },
            )
            for i, c in enumerate(items)
        ]
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(items)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(
                points=[_chunk_point_id(c) for c in ids]
            ),
            wait=True,
        )
        return len(ids)

    def delete_by_source(self, source_path: str) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_path",
                            match=qmodels.MatchValue(value=source_path),
                        ),
                        qmodels.FieldCondition(
                            key="kind",
                            match=qmodels.MatchValue(value="chunk"),
                        ),
                    ]
                )
            ),
            wait=True,
        )

    # ---- reads ----------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0 or not query or not query.strip():
            return []
        vec = _to_list(self.embedder.embed_one(query))
        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=k,
            query_filter=_chunk_only_filter(),
            with_payload=True,
        )
        out: list[tuple[str, float]] = []
        for point in result.points:
            payload = point.payload or {}
            cid = payload.get("chunk_id")
            if cid:
                out.append((cid, float(point.score)))
        return out

    def count(self) -> int:
        res = self._client.count(
            collection_name=self.collection_name,
            count_filter=_chunk_only_filter(),
            exact=True,
        )
        return int(res.count)


def _chunk_point_id(chunk_id: str) -> int:
    # Chunk IDs are 16 hex chars = 64 bits. Offset by 1 so nothing ever
    # collides with the reserved META_POINT_ID=0, even for all-zero hashes.
    return int(chunk_id, 16) + 1


def _to_list(vec) -> list[float]:
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


def _chunk_only_filter() -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="kind",
                match=qmodels.MatchValue(value="chunk"),
            )
        ]
    )
