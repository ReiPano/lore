# Hybrid Search Engine for Local Agents — Implementation Plan

## Goal

Build a local hybrid search system (BM25 + vector) that any MCP-compatible agent (Claude Desktop, Claude Code, etc.) can use as long-term memory and retrieval backend. Also expose a plain HTTP API so non-MCP agents can hit the same index.

## Success Criteria

- Index at least 10,000 documents/chunks locally with sub-200ms query latency.
- Hybrid search measurably outperforms pure vector search on a small eval set (queries with identifiers, proper nouns, rare terms).
- Works end-to-end from Claude Code or Claude Desktop via MCP.
- Index updates incrementally when source files change (no full rebuilds).
- Runs on a laptop, no GPU required.

## Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Best retrieval ecosystem |
| Vector store | Qdrant (Docker) | Fast, supports hybrid natively, good Python client |
| Lexical index | SQLite FTS5 | Zero-install, ships with Python, BM25 built in |
| Embeddings | FastEmbed (`bge-small-en-v1.5`) | Small, fast, runs on CPU, no external service |
| Reranker | `bge-reranker-base` via FastEmbed | Optional but big quality win |
| MCP server | FastMCP | Official Python SDK |
| HTTP API | FastAPI | Same process as MCP, minimal extra code |
| File watching | `watchdog` | Incremental reindexing |

## Repo Layout

```
hybrid-search/
├── pyproject.toml
├── README.md
├── docker-compose.yml          # Qdrant
├── config.yaml                 # paths to index, chunk size, etc.
├── src/
│   ├── hybrid_search/
│   │   ├── __init__.py
│   │   ├── chunking.py         # document → chunks
│   │   ├── embeddings.py       # FastEmbed wrapper
│   │   ├── lexical.py          # SQLite FTS5 BM25 index
│   │   ├── vector.py           # Qdrant client wrapper
│   │   ├── fusion.py           # RRF merge
│   │   ├── rerank.py           # cross-encoder reranking
│   │   ├── indexer.py          # end-to-end indexing pipeline
│   │   ├── search.py           # end-to-end query pipeline
│   │   ├── watcher.py          # file-change → reindex
│   │   ├── api.py              # FastAPI HTTP endpoints
│   │   └── mcp_server.py       # FastMCP server
├── tests/
│   ├── test_chunking.py
│   ├── test_fusion.py
│   ├── test_lexical.py
│   ├── test_vector.py
│   └── test_end_to_end.py
├── eval/
│   ├── queries.jsonl           # eval queries + expected doc IDs
│   └── run_eval.py             # compares bm25-only, vector-only, hybrid
└── scripts/
    ├── index_folder.py         # CLI: index a directory
    └── query.py                # CLI: one-off query
```

## Phase 0 — Project Setup

**Deliverable:** empty repo that installs cleanly, runs tests, and starts Qdrant.

- [ ] Initialize `pyproject.toml` with dependencies: `qdrant-client`, `fastembed`, `fastapi`, `uvicorn`, `fastmcp`, `watchdog`, `pyyaml`, `pytest`, `pytest-asyncio`.
- [ ] `docker-compose.yml` with a single Qdrant service on port 6333, volume-mounted storage.
- [ ] `config.yaml` with: `index_path`, `qdrant_url`, `collection_name`, `embedding_model`, `chunk_size`, `chunk_overlap`, `watch_paths`.
- [ ] `make up` / `make down` / `make test` targets (or the `just` equivalent).
- [ ] CI-style test that imports every module without error.

## Phase 1 — Chunking

**Deliverable:** given a file, produce a list of `Chunk` objects with stable IDs.

- [ ] Define `Chunk` dataclass: `id`, `source_path`, `text`, `start_offset`, `end_offset`, `metadata` (dict).
- [ ] `id` should be deterministic: hash of `(source_path, start_offset, text)` — lets us detect unchanged chunks on reindex.
- [ ] Implement three chunkers:
  - `TextChunker` — token-based, ~500 tokens, 50 overlap. Use `tiktoken` for token counts.
  - `MarkdownChunker` — splits on headings first, then falls back to token chunks for long sections.
  - `CodeChunker` — splits on function/class boundaries using `tree-sitter` for common languages (Python, JS, TS, Go, Rust). Falls back to line-count chunking for unsupported languages.
- [ ] Dispatcher picks chunker by file extension.
- [ ] Tests: chunker returns stable IDs, respects size limits, preserves offsets correctly.

**Gotcha:** don't split in the middle of code blocks in Markdown. The MarkdownChunker should treat fenced code as atomic.

## Phase 2 — Lexical Index (BM25 via SQLite FTS5)

**Deliverable:** `LexicalIndex` class with `add(chunks)`, `delete(chunk_ids)`, `search(query, k) -> [(chunk_id, score)]`.

- [ ] Schema: one FTS5 virtual table `chunks_fts(chunk_id UNINDEXED, text, source_path UNINDEXED)` plus a regular `chunks` table storing full metadata.
- [ ] Use FTS5's `bm25()` function for ranking. Default weights are fine to start.
- [ ] Tokenizer: use `unicode61` with `remove_diacritics 2` for general text; consider `porter` stemming for natural language corpora.
- [ ] Batch inserts in transactions (single-row inserts are slow).
- [ ] Test: insert 1000 chunks, query known terms, assert expected chunks come back in top 10.

**Gotcha:** FTS5 BM25 scores are negative (lower is better in SQLite's implementation). Negate them before fusion so higher = better.

## Phase 3 — Vector Index (Qdrant)

**Deliverable:** `VectorIndex` class with matching interface to `LexicalIndex`.

- [ ] On startup, ensure the Qdrant collection exists with the right vector size (384 for `bge-small-en-v1.5`).
- [ ] Use `FastEmbed` for embedding. Wrap in a class that batches (embed 32–64 chunks at a time).
- [ ] Store `chunk_id`, `source_path`, and a text preview in Qdrant payload — full text stays in SQLite.
- [ ] `search(query, k)` → embed query, Qdrant search, return `[(chunk_id, score)]`.
- [ ] `delete(chunk_ids)` via Qdrant's filter-based delete.
- [ ] Test: round-trip embedding + search returns the expected chunk for a query that paraphrases the chunk.

**Gotcha:** the embedding model must match between indexing and querying. Store the model name in collection metadata and refuse to query if it doesn't match config.

## Phase 4 — Fusion (RRF)

**Deliverable:** `reciprocal_rank_fusion(result_lists, k=60) -> merged_list`.

- [ ] Pure function, ~15 lines. Formula: `score(doc) = Σ 1 / (k + rank_in_list_i)`.
- [ ] `k=60` is the standard default and works well.
- [ ] Handle dedup: same chunk ID may appear in both lists.
- [ ] Support weighted fusion (optional parameter) — lets us tune BM25 vs vector influence later.
- [ ] Tests: chunk appearing high in both lists ranks above chunk appearing high in only one. Chunk appearing in only one list still surfaces.

## Phase 5 — End-to-End Search Pipeline

**Deliverable:** `HybridSearch.query(text, k=10) -> [SearchResult]`.

Pipeline:

1. Query both indexes in parallel (`asyncio.gather`) for top 50 each.
2. RRF merge to top 20.
3. (Optional, toggleable) rerank top 20 with cross-encoder, return top `k`.
4. Hydrate chunks: join IDs against SQLite for full text + metadata.

- [ ] `SearchResult` dataclass: `chunk_id`, `text`, `source_path`, `score`, `scores_breakdown` (dict showing BM25/vector/rerank contributions for debugging).
- [ ] Config flag to enable/disable reranking.
- [ ] Logging: log query, latencies per stage, top result IDs. Keep it structured (JSON) for later analysis.
- [ ] Test: on a fixture corpus, known queries return known expected chunks in top 3.

## Phase 6 — Indexing Pipeline

**Deliverable:** `Indexer.index_path(path)` that walks a directory and indexes everything.

- [ ] Supported file types: `.md`, `.txt`, `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.pdf`, `.docx`. Skip binaries and files over a size limit (configurable, default 10MB).
- [ ] Use the existing file-reading tools for PDFs (`pypdf` or similar) and docx (`python-docx`).
- [ ] Deduplicate: before inserting, check if chunk ID already exists. If so, skip.
- [ ] Track `source_path → [chunk_ids]` in SQLite so we can cleanly delete all chunks from a file.
- [ ] Parallelize: process files concurrently with a worker pool. Embedding is the bottleneck; keep batch sizes tuned.
- [ ] Progress bar with `rich` or `tqdm`.
- [ ] Tests: reindexing an unchanged directory is a no-op. Deleting a file removes its chunks.

## Phase 7 — File Watcher (Incremental Updates)

**Deliverable:** `watcher.py` runs in the background and keeps the index in sync.

- [ ] Use `watchdog` to subscribe to configured paths.
- [ ] On create/modify: re-chunk file, diff against existing chunk IDs, insert new, delete gone.
- [ ] On delete: remove all chunks for that path.
- [ ] Debounce: batch events within a 2-second window to avoid thrashing on rapid saves.
- [ ] Respect `.gitignore` if present (use `pathspec`).
- [ ] Tests: simulate file events and assert index state matches expectation.

## Phase 8 — HTTP API (FastAPI)

**Deliverable:** plain REST API so any agent framework can hit it.

Endpoints:

- `POST /search` — body: `{query, k, rerank}`, returns results.
- `POST /index` — body: `{paths: [...]}`, triggers indexing. Returns a job ID.
- `GET /jobs/{id}` — check indexing progress.
- `DELETE /documents` — body: `{paths: [...]}` or `{chunk_ids: [...]}`.
- `GET /health` — returns index stats (chunk count, last updated).

- [ ] Wrap in `uvicorn`, default to `localhost:8765`.
- [ ] Minimal auth: optional bearer token from config, off by default (local use).
- [ ] OpenAPI schema auto-generated by FastAPI → lets agents introspect the API.
- [ ] Tests with `httpx.AsyncClient`.

## Phase 9 — MCP Server

**Deliverable:** FastMCP server exposing the same functionality as MCP tools.

Tools to expose:

- `search(query: str, k: int = 10, rerank: bool = True) -> list[SearchResult]`
- `index_path(path: str) -> {status, chunks_added}`
- `list_sources() -> list[str]` — what's currently indexed
- `get_chunk(chunk_id: str) -> Chunk` — retrieve full chunk by ID (useful when the agent wants more context around a hit)

- [ ] Resources (MCP's read-only concept): expose indexed documents as resources so clients can enumerate them.
- [ ] Run over stdio for Claude Desktop; also support SSE for remote clients.
- [ ] Write a `claude_desktop_config.json` snippet in the README so users can copy-paste to connect.
- [ ] Test with `mcp-inspector` (Anthropic's debugging tool).

## Phase 10 — Evaluation

**Deliverable:** `eval/run_eval.py` that benchmarks bm25-only, vector-only, hybrid, and hybrid+rerank.

- [ ] Eval set format: `{query, relevant_chunk_ids: [...]}` in JSONL.
- [ ] Seed with 20–30 hand-written queries covering: natural language questions, keyword/identifier lookups, paraphrased concepts, multi-hop.
- [ ] Metrics: Recall@10, MRR, NDCG@10.
- [ ] Output a markdown table comparing the four configurations.
- [ ] Track eval scores in a committed `eval/results.md` so regressions are visible in PRs.

## Phase 11 — Polish

- [ ] README with: install, quickstart, architecture diagram, eval results, tuning guide.
- [ ] Example agent integrations:
  - Claude Desktop config snippet.
  - Claude Code: how to add the MCP server via `claude mcp add`.
  - LangChain: a `Retriever` subclass that hits the HTTP API.
  - Plain curl examples.
- [ ] Dockerfile for the whole app (not just Qdrant) — optional single-container deploy.
- [ ] Logging goes to both stdout and a rotating file.
- [ ] Graceful shutdown: flush SQLite, close Qdrant client.

## Out of Scope (for v1)

- Multi-user / multi-tenant indexes.
- GPU-accelerated embedding (document this as an upgrade path — swap FastEmbed for `sentence-transformers` with CUDA).
- Distributed Qdrant. Local single-node only.
- Query expansion / HyDE. Worth testing later but adds latency.
- Fine-tuning embeddings on your corpus. Big quality win eventually, but premature for v1.

## Order of Execution for Claude Code

Work through phases sequentially. Each phase should end with:

1. All tests for that phase passing.
2. A short entry in `CHANGELOG.md` describing what was built.
3. A commit with message `Phase N: <summary>`.

Do not skip ahead. Phase 5 (end-to-end) is the first point where the system is meaningfully useful — stopping there gives you a working CLI-based hybrid search. Phases 6–11 turn it into something agents can actually live against.

## Tuning Knobs to Expose in Config

Keep these in `config.yaml` from day one, even if defaults are fine. They're the things you'll want to adjust once you have real data:

- `chunk_size`, `chunk_overlap`
- `fusion_k` (RRF constant)
- `bm25_weight`, `vector_weight` (for weighted fusion)
- `rerank_top_n` (how many to rerank)
- `embedding_model`
- `retrieval_k_per_index` (top N pulled from each index before fusion)

## Non-Obvious Things Worth Getting Right Early

1. **Stable chunk IDs.** If you use auto-increment IDs, reindexing becomes a nightmare. Hash-based IDs let you diff cleanly.
2. **Store full text in SQLite, previews in Qdrant.** Qdrant payloads get bloated fast if you stuff full documents in them.
3. **Log the scores breakdown.** When hybrid results feel wrong, the first question is always "was BM25 or the vector side responsible?" — having that in logs saves hours.
4. **Async from the start.** Retrofitting async into a sync codebase is painful; FastAPI + FastMCP are async-native anyway.
5. **Make the reranker optional and measured.** It adds 50–200ms per query. On some corpora it's transformative; on others it barely helps. The eval script should make this visible.
