"""MCP server tests using FastMCP's in-memory client transport."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from hybrid_search.indexer import Indexer
from hybrid_search.lexical import LexicalIndex
from hybrid_search.mcp_server import create_server
from hybrid_search.search import HybridSearch
from hybrid_search.vector import VectorIndex

from .conftest import FakeEmbedder


@pytest.fixture
def stack(tmp_path: Path):
    lex = LexicalIndex(tmp_path / "lex.sqlite3")
    vec = VectorIndex.in_memory("mcp_test", FakeEmbedder(dim=32))
    indexer = Indexer(lex, vec, chunk_size=500, chunk_overlap=50)
    search = HybridSearch(
        lex,
        vec,
        rerank_enabled=False,
        retrieval_k_per_index=10,
        default_k=5,
    )
    yield lex, vec, indexer, search, tmp_path
    lex.close()
    vec.close()


def _unwrap(result) -> object:
    """FastMCP returns tool output under `.data` (structured) or `.content` (text)."""
    if getattr(result, "data", None) is not None:
        return result.data
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except Exception:  # noqa: BLE001
                return text
    return None


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


async def test_tools_registered(stack) -> None:
    _, _, indexer, search, _ = stack
    server = create_server(search=search, indexer=indexer)
    async with Client(server) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {"search", "index_path", "list_sources", "get_chunk"} <= names


async def test_index_path_and_list_sources(stack) -> None:
    _, _, indexer, search, tmp = stack
    _write(tmp, "a.md", "# Alpha\n\nHello world.")
    server = create_server(search=search, indexer=indexer)
    async with Client(server) as client:
        result = await client.call_tool("index_path", {"path": str(tmp)})
        stats = _unwrap(result)
        assert stats["files_indexed"] == 1
        assert stats["chunks_added"] > 0

        result = await client.call_tool("list_sources", {})
        sources = _unwrap(result)
        assert any("a.md" in s for s in sources)


async def test_search_tool_returns_hydrated_results(stack) -> None:
    lex, _, indexer, search, tmp = stack
    _write(tmp, "doc.md", "# Topic\n\nHybrid search blends lexical and semantic retrieval.")
    indexer.index_path(tmp)
    server = create_server(search=search, indexer=indexer)
    async with Client(server) as client:
        result = await client.call_tool(
            "search",
            {"query": "Hybrid search blends lexical and semantic retrieval."},
        )
        hits = _unwrap(result)
        assert hits
        top = hits[0]
        assert "chunk_id" in top
        assert "text" in top
        assert "scores_breakdown" in top


async def test_get_chunk_tool(stack) -> None:
    lex, _, indexer, search, tmp = stack
    target = _write(tmp, "doc.md", "# Title\n\nA chunk we can fetch by id.")
    indexer.index_file(target)
    ids = lex.chunk_ids_for_source(str(target.resolve()))
    assert ids

    server = create_server(search=search, indexer=indexer)
    async with Client(server) as client:
        result = await client.call_tool("get_chunk", {"chunk_id": ids[0]})
        chunk = _unwrap(result)
        assert chunk is not None
        assert chunk["id"] == ids[0]
        assert "chunk we can fetch" in chunk["text"]

        # Missing chunk returns None-ish.
        result = await client.call_tool("get_chunk", {"chunk_id": "deadbeefdeadbeef"})
        assert _unwrap(result) is None


async def test_resources_registered_and_readable(stack) -> None:
    _, _, indexer, search, tmp = stack
    _write(tmp, "alpha.md", "# Alpha\n\nFirst source text.")
    indexer.index_path(tmp)
    server = create_server(search=search, indexer=indexer)
    async with Client(server) as client:
        resources = await client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "lore://sources" in uris

        content = await client.read_resource("lore://sources")
        # Content is a list of TextResourceContents.
        text_blocks = [getattr(c, "text", "") for c in content]
        combined = "\n".join(text_blocks)
        assert "alpha.md" in combined
