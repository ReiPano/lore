"""FastMCP server exposing hybrid search as MCP tools and resources.

``create_server`` wires a ``HybridSearch`` and an ``Indexer`` into a
``FastMCP`` instance. The result can be run over stdio (for Claude Desktop
and Claude Code via ``claude mcp add``) or mounted into an HTTP transport.

Tools
-----
- ``search(query, k, rerank, project)`` — hybrid search with optional
  project-scoped filter.
- ``search_in_file(path, query, k, rerank)`` — scope search to one file or
  directory.
- ``related(chunk_id, k, project)`` — nearest-neighbour chunks.
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
from .projects import ProjectStore
from .search import HybridSearch

log = logging.getLogger(__name__)


def resolve_project_prefix(
    project: str | None,
    *,
    project_store: ProjectStore | None,
) -> str | None:
    """Turn a project name (or raw path) into an absolute-path prefix.

    Strings that don't match a registered project name are treated as raw
    path prefixes. This lets callers pass either a human-friendly name or an
    explicit path filter.
    """
    if not project:
        return None
    if project_store is not None:
        hit = project_store.find(project)
        if hit is not None:
            return hit.path
    return project


def create_server(
    *,
    search: HybridSearch,
    indexer: Indexer,
    project_store: ProjectStore | None = None,
    name: str = "lore",
) -> FastMCP:
    mcp = FastMCP(name)

    @mcp.tool(name="search")
    async def search_tool(
        query: str,
        k: int = 10,
        rerank: bool | None = None,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + vector search over the local memory.

        Use ``project`` to scope the query to one indexed project by name
        (see ``list_sources``). Leave it empty to search across everything.
        ``rerank=True`` polishes the top candidates with a cross-encoder —
        set it on unless the caller is latency-sensitive.
        """
        source_prefix = resolve_project_prefix(project, project_store=project_store)
        results = await search.aquery(
            query,
            k,
            rerank=rerank,
            source_prefix=source_prefix,
        )
        return [_result_to_dict(r) for r in results]

    @mcp.tool(name="search_in_file")
    async def search_in_file_tool(
        path: str,
        query: str,
        k: int = 10,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Search within a single file or directory.

        ``path`` is matched as an absolute-path prefix, so passing a folder
        scopes the search to everything underneath it. Aggregation is
        disabled so callers can see multiple chunks from the same file.
        """
        results = await search.asearch_in_file(path, query, k, rerank=rerank)
        return [_result_to_dict(r) for r in results]

    @mcp.tool(name="related")
    async def related_tool(
        chunk_id: str,
        k: int = 10,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks semantically similar to ``chunk_id``.

        Useful for "find more like this" follow-ups after a ``search`` or
        ``get_chunk`` hit. Results exclude the seed chunk itself.
        """
        source_prefix = resolve_project_prefix(project, project_store=project_store)
        results = await search.arelated(chunk_id, k, source_prefix=source_prefix)
        return [_result_to_dict(r) for r in results]

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


def _result_to_dict(r) -> dict[str, Any]:
    return {
        "chunk_id": r.chunk_id,
        "text": r.text,
        "source_path": r.source_path,
        "score": r.score,
        "scores_breakdown": dict(r.scores_breakdown),
    }


def run_stdio(
    *,
    search: HybridSearch,
    indexer: Indexer,
    project_store: ProjectStore | None = None,
) -> None:
    create_server(
        search=search, indexer=indexer, project_store=project_store
    ).run()
