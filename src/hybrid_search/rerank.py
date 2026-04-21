"""Cross-encoder reranker wrapper.

Uses FastEmbed's ``TextCrossEncoder``. Model download is deferred until the
first ``score`` call so importing this module never hits the network.
"""

from __future__ import annotations

DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Lazy cross-encoder wrapper.

    The default model is small and ships through FastEmbed; the plan targets
    ``bge-reranker-base`` but that weight file is large. Callers can override
    ``model_name`` at construction.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._ensure_model()
        # FastEmbed's TextCrossEncoder exposes `rerank(query, documents)` as a
        # generator of relevance scores (higher = better).
        raw = list(model.rerank(query=query, documents=list(texts)))
        return [float(s) for s in raw]

    def _ensure_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
            )
        return self._model
