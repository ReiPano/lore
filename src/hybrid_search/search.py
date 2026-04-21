"""End-to-end hybrid search pipeline.

Retrieval → fusion → optional rerank → hydrate. The two retrievers run in
parallel worker threads so a slow embedding call cannot block BM25. The
``scores_breakdown`` on each result preserves the component signals (bm25,
vector, rrf, rerank) so debugging and eval runs can tell which side of the
hybrid won.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from .fusion import reciprocal_rank_fusion
from .lexical import LexicalIndex
from .vector import VectorIndex

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    text: str
    source_path: str
    score: float
    scores_breakdown: dict[str, float] = field(default_factory=dict)


class RerankerProtocol(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class HybridSearch:
    def __init__(
        self,
        lexical: LexicalIndex,
        vector: VectorIndex,
        *,
        reranker: RerankerProtocol | None = None,
        retrieval_k_per_index: int = 50,
        fusion_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        rerank_top_n: int = 20,
        default_k: int = 10,
        rerank_enabled: bool = True,
        aggregate_by_file: bool = True,
        max_per_file: int = 1,
    ) -> None:
        self.lexical = lexical
        self.vector = vector
        self.reranker = reranker
        self.retrieval_k_per_index = retrieval_k_per_index
        self.fusion_k = fusion_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rerank_top_n = rerank_top_n
        self.default_k = default_k
        self.rerank_enabled = rerank_enabled
        self.aggregate_by_file = aggregate_by_file
        self.max_per_file = max(1, int(max_per_file))

    # ---- public API ---------------------------------------------------

    def query(
        self,
        query: str,
        k: int | None = None,
        *,
        rerank: bool | None = None,
        aggregate_by_file: bool | None = None,
    ) -> list[SearchResult]:
        return asyncio.run(
            self.aquery(query, k, rerank=rerank, aggregate_by_file=aggregate_by_file)
        )

    async def aquery(
        self,
        query: str,
        k: int | None = None,
        *,
        rerank: bool | None = None,
        aggregate_by_file: bool | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        limit = k if k is not None else self.default_k
        if limit <= 0:
            return []

        do_rerank = rerank if rerank is not None else self.rerank_enabled
        do_rerank = bool(do_rerank and self.reranker is not None)

        t_start = time.perf_counter()

        bm25_task = asyncio.to_thread(
            self.lexical.search, query, self.retrieval_k_per_index
        )
        vec_task = asyncio.to_thread(
            self.vector.search, query, self.retrieval_k_per_index
        )
        bm25_hits, vec_hits = await asyncio.gather(bm25_task, vec_task)

        t_retrieved = time.perf_counter()

        bm25_scores = dict(bm25_hits)
        vec_scores = dict(vec_hits)

        fused = reciprocal_rank_fusion(
            [bm25_hits, vec_hits],
            k=self.fusion_k,
            weights=[self.bm25_weight, self.vector_weight],
            top_n=max(limit, self.rerank_top_n),
        )
        fused_scores = dict(fused)

        t_fused = time.perf_counter()

        top_ids = [cid for cid, _ in fused]
        chunks_by_id = {c.id: c for c in self.lexical.get_many(top_ids)}

        rerank_scores: dict[str, float] = {}
        ordered: list[tuple[str, float]]
        if do_rerank and fused:
            assert self.reranker is not None  # for type checkers
            rr_slice = [cid for cid, _ in fused[: self.rerank_top_n] if cid in chunks_by_id]
            rr_texts = [chunks_by_id[cid].text for cid in rr_slice]
            raw_scores = self.reranker.score(query, rr_texts)
            rerank_scores = {cid: float(s) for cid, s in zip(rr_slice, raw_scores)}
            reranked = sorted(rr_slice, key=lambda cid: -rerank_scores[cid])
            tail = [(cid, fused_scores[cid]) for cid, _ in fused[self.rerank_top_n :]]
            ordered = [(cid, rerank_scores[cid]) for cid in reranked] + tail
        else:
            ordered = fused

        t_reranked = time.perf_counter()

        do_aggregate = (
            aggregate_by_file if aggregate_by_file is not None else self.aggregate_by_file
        )
        if do_aggregate and ordered:
            ordered = _dedupe_by_source(
                ordered,
                chunks_by_id,
                max_per_file=self.max_per_file,
            )

        results: list[SearchResult] = []
        for doc_id, _ in ordered[:limit]:
            chunk = chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            breakdown: dict[str, float] = {}
            if doc_id in bm25_scores:
                breakdown["bm25"] = bm25_scores[doc_id]
            if doc_id in vec_scores:
                breakdown["vector"] = vec_scores[doc_id]
            if doc_id in fused_scores:
                breakdown["rrf"] = fused_scores[doc_id]
            if doc_id in rerank_scores:
                breakdown["rerank"] = rerank_scores[doc_id]
            final_score = (
                rerank_scores[doc_id]
                if doc_id in rerank_scores
                else fused_scores.get(doc_id, 0.0)
            )
            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    text=chunk.text,
                    source_path=chunk.source_path,
                    score=final_score,
                    scores_breakdown=breakdown,
                )
            )

        t_done = time.perf_counter()
        log.info(
            json.dumps(
                {
                    "event": "hybrid_search",
                    "query": query,
                    "k": limit,
                    "rerank": do_rerank,
                    "aggregate_by_file": do_aggregate,
                    "bm25_hits": len(bm25_hits),
                    "vector_hits": len(vec_hits),
                    "fused_hits": len(fused),
                    "returned": len(results),
                    "top_ids": [r.chunk_id for r in results],
                    "latency_ms": {
                        "retrieve": round((t_retrieved - t_start) * 1000, 2),
                        "fuse": round((t_fused - t_retrieved) * 1000, 2),
                        "rerank": round((t_reranked - t_fused) * 1000, 2),
                        "hydrate": round((t_done - t_reranked) * 1000, 2),
                        "total": round((t_done - t_start) * 1000, 2),
                    },
                }
            )
        )
        return results


def _dedupe_by_source(
    ordered: list[tuple[str, float]],
    chunks_by_id: dict,
    *,
    max_per_file: int,
) -> list[tuple[str, float]]:
    """Two-pass file diversification.

    Pass 1 walks ``ordered`` in score order and keeps the first
    ``max_per_file`` chunks per ``source_path``. Pass 2 appends everything
    that did not make the first cut so the caller can still backfill if it
    needs more results than unique sources. Chunks whose hydration lookup
    failed (not in ``chunks_by_id``) are kept in their original slot — we
    can't group what we can't identify.
    """
    primary: list[tuple[str, float]] = []
    overflow: list[tuple[str, float]] = []
    per_source: dict[str, int] = {}
    for entry in ordered:
        doc_id, _ = entry
        chunk = chunks_by_id.get(doc_id)
        if chunk is None:
            # Can't group; keep it in the primary list at its natural position.
            primary.append(entry)
            continue
        source = chunk.source_path
        count = per_source.get(source, 0)
        if count < max_per_file:
            primary.append(entry)
            per_source[source] = count + 1
        else:
            overflow.append(entry)
    return primary + overflow
