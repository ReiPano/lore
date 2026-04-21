"""Reciprocal rank fusion.

Pure function. Each input list is ordered best-first; items may be bare IDs or
``(id, score)`` tuples (scores are ignored — only ranks matter). Items repeat
across lists contribute additively, so a document that ranks high in both
indexes beats one that only wins in one.

Score formula::

    score(doc) = sum(weight_i / (k + rank_in_list_i))

where ``rank`` is 1-based.
"""

from __future__ import annotations

from typing import Iterable, Sequence

DEFAULT_K = 60

ResultItem = str | tuple[str, float]


def reciprocal_rank_fusion(
    result_lists: Sequence[Iterable[ResultItem]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    lists = [list(lst) for lst in result_lists]
    if not lists:
        return []

    if weights is None:
        effective_weights = [1.0] * len(lists)
    else:
        if len(weights) != len(lists):
            raise ValueError(
                f"weights length {len(weights)} does not match result_lists length {len(lists)}"
            )
        effective_weights = [float(w) for w in weights]

    fused: dict[str, float] = {}
    for lst, weight in zip(lists, effective_weights):
        if weight == 0.0:
            continue
        seen_in_list: set[str] = set()
        for rank, item in enumerate(lst, start=1):
            doc_id = _item_id(item)
            if doc_id in seen_in_list:
                # Same list shouldn't double-count a document.
                continue
            seen_in_list.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return ordered


def _item_id(item: ResultItem) -> str:
    if isinstance(item, tuple):
        return item[0]
    return item
