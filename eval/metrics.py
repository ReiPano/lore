"""Eval metrics for ranked retrieval.

Relevance is binary: a returned chunk is "relevant" iff any substring from the
query's ``must_contain`` list appears (case-insensitive) in the chunk's text.
This keeps the eval resilient to chunker changes — we care about whether the
right region of the corpus surfaced, not which exact chunk ID won.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: str
    retrieved_texts: list[str]
    must_contain: list[str]

    def relevant_ranks(self) -> list[int]:
        out: list[int] = []
        for rank, text in enumerate(self.retrieved_texts, start=1):
            if is_relevant(text, self.must_contain):
                out.append(rank)
        return out


def is_relevant(text: str, must_contain: list[str]) -> bool:
    if not must_contain:
        return False
    lowered = text.lower()
    return any(s.lower() in lowered for s in must_contain)


def hit_at_k(ranks: list[int], k: int) -> float:
    return 1.0 if any(r <= k for r in ranks) else 0.0


def mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return 1.0 / float(min(ranks))


def ndcg_at_k(ranks: list[int], k: int) -> float:
    """Binary NDCG@k with an ideal ranking of a single relevant doc at rank 1."""
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    idcg = 1.0 / math.log2(2)  # 1.0 — relevant doc placed at rank 1
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True, slots=True)
class Aggregate:
    hit_at_10: float
    mrr: float
    ndcg_at_10: float
    n_queries: int

    def as_dict(self) -> dict[str, float]:
        return {
            "hit@10": round(self.hit_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "n": float(self.n_queries),
        }


def aggregate(results: list[QueryResult], *, k: int = 10) -> Aggregate:
    if not results:
        return Aggregate(0.0, 0.0, 0.0, 0)
    hit_scores = []
    mrr_scores = []
    ndcg_scores = []
    for r in results:
        ranks = r.relevant_ranks()
        hit_scores.append(hit_at_k(ranks, k))
        mrr_scores.append(mrr([x for x in ranks if x <= k]))
        ndcg_scores.append(ndcg_at_k(ranks, k))
    n = len(results)
    return Aggregate(
        hit_at_10=sum(hit_scores) / n,
        mrr=sum(mrr_scores) / n,
        ndcg_at_10=sum(ndcg_scores) / n,
        n_queries=n,
    )
