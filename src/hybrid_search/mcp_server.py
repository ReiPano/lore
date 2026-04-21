"""FastMCP server exposing hybrid search as MCP tools and resources.

``create_server`` wires a ``HybridSearch`` and an ``Indexer`` into a
``FastMCP`` instance. The result can be run over stdio (for Claude Desktop
and Claude Code via ``claude mcp add``) or mounted into an HTTP transport.

Tools
-----
- ``search(query, k, rerank)`` — hybrid search, returns hydrated results.
- ``index_path(path)`` — walk a directory or index a single file.
- ``list_sources()`` — all source paths currently indexed.
- ``get_chunk(chunk_id)`` — fetch a chunk by ID with full text and metadata.

Resources
---------
- ``lore://sources`` — newline-separated list of indexed sources.
- ``lore://source/{path}`` — concatenated chunk text for one source.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from .indexer import Indexer
from .search import HybridSearch

log = logging.getLogger(__name__)


def create_server(
    *,
    search: HybridSearch,
    indexer: Indexer,
    name: str = "lore",
) -> FastMCP:
    mcp = FastMCP(name)

    @mcp.tool(name="search")
    async def search_tool(
        query: str,
        k: int = 10,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search over the local BM25 + vector index."""
        results = await search.aquery(query, k, rerank=rerank)
        return [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "source_path": r.source_path,
                "score": r.score,
                "scores_breakdown": dict(r.scores_breakdown),
            }
            for r in results
        ]

    @mcp.tool(name="index_path")
    def index_path_tool(path: str) -> dict[str, Any]:
        """Index a directory (recursive) or a single file. Returns stats."""
        return asdict(indexer.index_path(path))

    @mcp.tool(name="list_sources")
    def list_sources_tool() -> list[str]:
        """Return all source paths currently indexed."""
        return indexer.lexical.sources()

    @mcp.tool(name="get_chunk")
    def get_chunk_tool(chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a chunk by ID with full text, offsets, and metadata."""
        chunk = indexer.lexical.get(chunk_id)
        if chunk is None:
            return None
        return {
            "id": chunk.id,
            "source_path": chunk.source_path,
            "text": chunk.text,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "metadata": dict(chunk.metadata),
        }

    @mcp.resource("lore://sources")
    def sources_resource() -> str:
        """Newline-separated list of indexed sources."""
        return "\n".join(indexer.lexical.sources())

    @mcp.resource("lore://source/{path}")
    def source_resource(path: str) -> str:
        """Full reassembled text for one source, chunks joined in offset order."""
        ids = indexer.lexical.chunk_ids_for_source(path)
        chunks = indexer.lexical.get_many(ids)
        return "\n\n".join(c.text for c in chunks)

    return mcp


def run_stdio(*, search: HybridSearch, indexer: Indexer) -> None:
    create_server(search=search, indexer=indexer).run()
