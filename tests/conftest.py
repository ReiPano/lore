"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic hash-based embedder used by unit tests.

    Produces a normalized vector per input. Identical strings yield identical
    vectors, so nearest-neighbor search for a stored chunk's own text returns
    that chunk. This is enough to exercise storage/retrieval plumbing without
    downloading real models.
    """

    model_name = "fake-hash-v1"

    def __init__(self, dim: int = 16) -> None:
        if dim < 4:
            raise ValueError("dim must be >= 4")
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        # SHA-256 gives 32 deterministic bytes; stretch or slice to `dim`.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = bytearray()
        while len(raw) < self.dim:
            raw.extend(digest)
            digest = hashlib.sha256(digest).digest()
        arr = np.frombuffer(bytes(raw[: self.dim]), dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm else arr


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("HYBRID_RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test (set HYBRID_RUN_INTEGRATION=1 to run)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
