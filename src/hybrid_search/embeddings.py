"""FastEmbed wrapper with batching and lazy model load."""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH = 32


class Embedder:
    """Lazy wrapper around ``fastembed.TextEmbedding``.

    The model is downloaded on first embed call, not at construction, so
    importing this module never touches the network.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = DEFAULT_BATCH,
        cache_dir: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache_dir = cache_dir
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            model = self._ensure_model()
            vec = next(iter(model.embed(["_"])))
            self._dim = int(len(vec))
        return self._dim

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        items = list(texts)
        if not items:
            return []
        model = self._ensure_model()
        out: list[np.ndarray] = []
        for batch in _batched(items, self.batch_size):
            out.extend(np.asarray(v, dtype=np.float32) for v in model.embed(batch))
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
            )
        return self._model


def _batched(seq: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
