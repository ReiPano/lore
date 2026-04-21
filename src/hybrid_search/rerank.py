"""Cross-encoder reranker wrapper.

Uses FastEmbed's ``TextCrossEncoder``. Model download is deferred until the
first ``score`` call so importing this module never hits the network.
"""

from __future__ import annotations

import logging

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Tried in order if the primary model cannot load (network error, missing
# weights in the user's FastEmbed version). Keeps the pipeline functional
# instead of crashing a whole query.
FALLBACK_RERANKER_MODELS: tuple[str, ...] = (
    "BAAI/bge-reranker-base",
    "Xenova/ms-marco-MiniLM-L-6-v2",
)

log = logging.getLogger(__name__)


class Reranker:
    """Lazy cross-encoder wrapper with graceful fallback.

    The default is ``bge-reranker-v2-m3`` (multilingual, ~550 MB). If that
    model isn't available the wrapper silently drops down to the older
    ``bge-reranker-base`` and then the tiny MS-MARCO MiniLM — whichever
    loads first wins for the lifetime of the instance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        cache_dir: str | None = None,
        fallbacks: tuple[str, ...] | None = None,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._fallbacks = fallbacks if fallbacks is not None else FALLBACK_RERANKER_MODELS
        self._model = None
        self._active_model_name = model_name

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._ensure_model()
        # FastEmbed's TextCrossEncoder exposes `rerank(query, documents)` as a
        # generator of relevance scores (higher = better).
        raw = list(model.rerank(query=query, documents=list(texts)))
        return [float(s) for s in raw]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        candidates = (self.model_name, *self._fallbacks)
        last_exc: Exception | None = None
        for name in candidates:
            try:
                self._model = TextCrossEncoder(
                    model_name=name,
                    cache_dir=self._cache_dir,
                )
                self._active_model_name = name
                if name != self.model_name:
                    log.warning(
                        "reranker model %r unavailable; using fallback %r",
                        self.model_name,
                        name,
                    )
                return self._model
            except Exception as exc:  # noqa: BLE001 - try next fallback
                last_exc = exc
                log.debug("reranker %r failed to load: %s", name, exc)
        raise RuntimeError(
            f"could not load any reranker model (tried: {list(candidates)})"
        ) from last_exc
