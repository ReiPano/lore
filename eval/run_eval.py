"""Benchmark BM25, vector, hybrid, and hybrid+rerank on a small eval set.

Usage (after ``make install``):

    .venv/bin/python -m eval.run_eval
    .venv/bin/python -m eval.run_eval --fake-embedder      # quick smoke run
    .venv/bin/python -m eval.run_eval --include-rerank     # downloads reranker

The eval corpus lives in ``eval/corpus/`` and the query set in
``eval/queries.jsonl``. Results land in ``eval/results.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from hybrid_search.embeddings import DEFAULT_MODEL, Embedder
from hybrid_search.indexer import Indexer
from hybrid_search.lexical import LexicalIndex
from hybrid_search.search import HybridSearch
from hybrid_search.vector import VectorIndex

from .metrics import Aggregate, QueryResult, aggregate

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_QUERIES = ROOT / "queries.jsonl"
DEFAULT_RESULTS = ROOT / "results.md"


@dataclass(slots=True)
class EvalQuery:
    query: str
    must_contain: list[str]
    notes: str = ""


def load_queries(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        queries.append(
            EvalQuery(
                query=obj["query"],
                must_contain=list(obj["must_contain"]),
                notes=obj.get("notes", ""),
            )
        )
    return queries


RetrieveFn = Callable[[str, int], Awaitable[list[str]]]


async def _retrieve_lexical(lex: LexicalIndex, query: str, k: int) -> list[str]:
    hits = lex.search(query, k)
    ids = [cid for cid, _ in hits]
    chunks = lex.get_many(ids)
    by_id = {c.id: c for c in chunks}
    return [by_id[cid].text for cid in ids if cid in by_id]


async def _retrieve_vector(vec: VectorIndex, lex: LexicalIndex, query: str, k: int) -> list[str]:
    hits = vec.search(query, k)
    ids = [cid for cid, _ in hits]
    chunks = lex.get_many(ids)
    by_id = {c.id: c for c in chunks}
    return [by_id[cid].text for cid in ids if cid in by_id]


async def _retrieve_hybrid(search: HybridSearch, query: str, k: int, *, rerank: bool) -> list[str]:
    results = await search.aquery(query, k, rerank=rerank)
    return [r.text for r in results]


async def run_eval(
    corpus_dir: Path,
    queries: list[EvalQuery],
    *,
    embedder,
    include_rerank: bool = False,
    k: int = 10,
    reranker=None,
) -> dict[str, tuple[Aggregate, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        lex = LexicalIndex(Path(tmp) / "lex.sqlite3")
        vec = VectorIndex.in_memory("eval", embedder)
        try:
            indexer = Indexer(lex, vec, chunk_size=500, chunk_overlap=50)
            indexer.index_path(corpus_dir)
            hybrid_no_rerank = HybridSearch(
                lex,
                vec,
                rerank_enabled=False,
                retrieval_k_per_index=max(k * 2, 20),
                default_k=k,
            )
            configs: dict[str, RetrieveFn] = {
                "bm25": lambda q, kk: _retrieve_lexical(lex, q, kk),
                "vector": lambda q, kk: _retrieve_vector(vec, lex, q, kk),
                "hybrid": lambda q, kk: _retrieve_hybrid(hybrid_no_rerank, q, kk, rerank=False),
            }
            if include_rerank and reranker is not None:
                hybrid_rerank = HybridSearch(
                    lex,
                    vec,
                    reranker=reranker,
                    rerank_enabled=True,
                    retrieval_k_per_index=max(k * 2, 20),
                    rerank_top_n=max(k, 20),
                    default_k=k,
                )
                configs["hybrid+rerank"] = lambda q, kk: _retrieve_hybrid(
                    hybrid_rerank, q, kk, rerank=True
                )

            out: dict[str, tuple[Aggregate, float]] = {}
            for name, retrieve in configs.items():
                t0 = time.perf_counter()
                per_query: list[QueryResult] = []
                for q in queries:
                    texts = await retrieve(q.query, k)
                    per_query.append(
                        QueryResult(
                            query=q.query,
                            retrieved_texts=texts,
                            must_contain=q.must_contain,
                        )
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                out[name] = (aggregate(per_query, k=k), elapsed_ms)
            return out
        finally:
            lex.close()
            vec.close()


def format_markdown(
    results: dict[str, tuple[Aggregate, float]],
    *,
    embedder_name: str,
    corpus: Path,
    queries: Path,
    k: int,
) -> str:
    lines = [
        "# Hybrid Search Eval",
        "",
        f"- corpus: `{corpus}`",
        f"- queries: `{queries}`",
        f"- embedder: `{embedder_name}`",
        f"- k: {k}",
        "",
        "| config | hit@k | mrr | ndcg@k | total ms |",
        "|---|---|---|---|---|",
    ]
    for name, (agg, ms) in results.items():
        lines.append(
            f"| {name} | {agg.hit_at_10:.3f} | {agg.mrr:.3f} | {agg.ndcg_at_10:.3f} | {ms:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark hybrid search configs.")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use the in-process hash embedder (quick smoke run, no quality).",
    )
    p.add_argument(
        "--include-rerank",
        action="store_true",
        help="Also evaluate hybrid+cross-encoder rerank (downloads reranker).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    return p.parse_args(argv)


def _build_embedder(args: argparse.Namespace) -> tuple[Any, str]:
    if args.fake_embedder:
        # Import locally so the standalone CLI doesn't require pytest at runtime.
        from tests.conftest import FakeEmbedder

        return FakeEmbedder(dim=32), "fake-hash-v1"
    return Embedder(model_name=args.model), args.model


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queries = load_queries(args.queries)
    embedder, embedder_name = _build_embedder(args)
    reranker = None
    if args.include_rerank:
        from hybrid_search.rerank import Reranker

        reranker = Reranker()
    results = asyncio.run(
        run_eval(
            args.corpus,
            queries,
            embedder=embedder,
            include_rerank=args.include_rerank,
            k=args.k,
            reranker=reranker,
        )
    )
    table = format_markdown(
        results,
        embedder_name=embedder_name,
        corpus=args.corpus,
        queries=args.queries,
        k=args.k,
    )
    print(table)
    args.output.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
