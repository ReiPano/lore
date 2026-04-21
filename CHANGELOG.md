# Changelog

## 0.2.0 — Milestone 1 (polish, security, CI)

- **File-level hit aggregation.** `HybridSearch.aquery` now diversifies results by `source_path` so one hot file cannot monopolize the top-k. Enabled by default (`aggregate_by_file=True`); override per-call or disable in the constructor. New helper `_dedupe_by_source` in `search.py`.
- **Linear-time TextChunker.** Rewrote `TextChunker.split` with a byte-offset map built via `enc.decode_single_token_bytes`. Replaces the previous O(n²) prefix-decode loop. Multi-megabyte files now chunk in well under a second; added regression tests covering UTF-8 boundaries and a 200k-token synthetic input.
- **Sensitive-file defaults.** `config.yaml` ships an expanded `exclude_patterns` list (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `credentials.json`, `.npmrc`, `.pypirc`, etc.) plus a new `exclude_content_patterns` regex list (Anthropic, OpenAI project, AWS, GitHub, Slack tokens, PEM private-key headers). Indexer scans the first 32 KB of each candidate file and skips it if any pattern matches; records `files_skipped_sensitive` on `IndexStats`.
- **GitHub Actions CI.** Added `.github/workflows/ci.yml`. Runs `ruff check` + `pytest -q` on pushes and PRs across Python 3.11 / 3.12 (Ubuntu) and Python 3.12 (macOS).
- **Tests.** Added `tests/test_config.py` plus new cases in `test_search.py`, `test_chunking.py`, `test_indexer.py` covering aggregation, chunker invariants, sensitive-file skip by filename and by content, and config loading with the new field. Suite now at 157 tests.
- README "Exclusions" section documents the new defaults.

## Unreleased

### Phase 12 — Unified CLI
- Added `ProjectStore` (`projects.py`) persisting named paths under `.data/projects.json` with dedup-by-path and unique-name generation.
- Added lifecycle helpers (`lifecycle.py`): `qdrant_up`/`qdrant_down`/`qdrant_status` via `docker compose`, plus `watcher_start`/`watcher_stop` / `watcher_running` with pidfile + detached process.
- Added `__main__.py` so `python -m hybrid_search ...` works (used by the spawned watcher daemon).
- Rewrote the CLI with subcommands: `up`, `down`, `status`, `projects list|add|remove|reindex`, `tui` (interactive menu via `rich`), plus existing `index`, `query`, `serve-api`, `serve-mcp`, `watch`.
- Added `tests/test_projects.py` (10 tests) for the store.

### Phase 11 — Polish
- Added `logging_setup.configure_logging` with stdout + rotating-file handlers.
- Rewrote `cli.py` as a multi-command entry point: `index`, `query`, `serve-api`, `serve-mcp`, `watch`. Exposed via `hybrid-search` console script; legacy `hybrid-index` / `hybrid-query` kept as thin shims.
- Added `Dockerfile` for a single-container deployment of the FastAPI server.
- Added `examples/claude_desktop_config.json` and `examples/curl.sh` so users can wire Claude Desktop or drive the HTTP API without reading the source.
- Rewrote `README.md` with quickstart, architecture diagram, MCP / HTTP / LangChain integration notes, eval instructions, tuning guide.

### Phase 10 — Evaluation
- Added `eval/metrics.py`: `is_relevant`, `hit_at_k`, `mrr`, `ndcg_at_k`, `aggregate`. Binary relevance via case-insensitive substring matching on `must_contain`.
- Added `eval/run_eval.py`: runner that benchmarks BM25, vector, hybrid, and optional hybrid+rerank configs; writes a markdown results table and supports `--fake-embedder` for smoke runs.
- Added `eval/corpus/*.md` (8 docs) and `eval/queries.jsonl` (10 queries).
- Added `tests/test_eval.py` (7 tests): metric correctness, query-file loader, full pipeline smoke run.

### Phase 9 — MCP Server
- Added `create_server(search, indexer, name)` + `run_stdio` helpers in `mcp_server.py` (FastMCP 3).
- Tools: `search`, `index_path`, `list_sources`, `get_chunk`.
- Resources: `hybrid-search://sources` (list) and `hybrid-search://source/{path}` (reassembled text).
- Added `tests/test_mcp_server.py` (5 tests) using FastMCP's in-memory `Client` transport: tool discovery, index_path + list_sources, search hydration, get_chunk hit/miss, resource read.

### Phase 8 — HTTP API
- Added `create_app(search, indexer, *, auth_token)` factory in `api.py` returning a `FastAPI` app with pydantic-validated requests/responses.
- Endpoints: `POST /search`, `POST /index` (returns job_id), `GET /jobs/{id}`, `DELETE /documents` (paths or chunk_ids), `GET /health`.
- Background indexing: `asyncio.create_task` + `asyncio.to_thread` fan out each path, reports status + merged `IndexStats` through the job registry.
- Optional bearer-token dependency guards every endpoint when `auth_token` is set; off by default for local use.
- Added `tests/test_api.py` (10 tests) exercising health, indexing (with poll), search round-trip, delete by path + by chunk id, empty-delete guard, missing-job 404, auth required/accepted, job-error surfacing.

### Phase 7 — File Watcher
- Added `IndexWatcher` (`watcher.py`): `watchdog.Observer` feeds `on_upsert`/`on_delete` hooks which coalesce pending paths; a background thread debounces within a configurable window (default 2s) and calls `Indexer.index_file` / `Indexer.remove_path`.
- Filters pruned directories (`PRUNED_DIRS` now public on indexer module) and, when enabled, entries matched by a top-level `.gitignore` via `pathspec` (prefers the non-deprecated `gitignore` style).
- Idempotent `start()`/`stop()`, context-manager support, `flush()` for synchronous drain, `has_pending()` for tests.
- Added `tests/test_watcher.py` (10 tests): upsert/flush, delete, dedup under rapid saves, delete→upsert ordering, upsert→delete ordering, pruned-dir guard, gitignore opt-in / opt-out, async worker flush, lifecycle safety.

### Phase 6 — Indexing Pipeline
- Added `Indexer` + `IndexStats` (`indexer.py`): walks a directory, reads each supported file, diffs new chunk IDs against existing ones for that source, and writes only the delta so reindexing is cheap.
- Prunes common noise dirs (`.git`, `node_modules`, `__pycache__`, `.venv`, `dist`, `build`, `.data`, caches).
- Optional PDF / DOCX readers (`pypdf`, `python-docx`) under the `parsers` extras; files are skipped gracefully with a warning when the parser package is missing.
- `remove_path` supports individual files and whole directories; deletes from both indexes before returning.
- Best-effort cross-index rollback if the second write fails after the first succeeded.
- Optional `rich.Progress` bar when `show_progress=True`.
- Added `tests/test_indexer.py` (10 tests): single-file, recursive walk with pruning, idempotent reindex, edit detection, per-file and per-directory removal, size-limit skip, unsupported-extension skip, parser-missing fallback, missing-path error.

### Phase 5 — End-to-End Search
- Added `Reranker` (`rerank.py`): lazy wrapper around FastEmbed `TextCrossEncoder`.
- Added `HybridSearch` + `SearchResult` (`search.py`): parallel BM25/vector retrieval via `asyncio.to_thread` + `asyncio.gather`, RRF fusion, optional cross-encoder rerank, chunk hydration through `LexicalIndex.get_many`.
- `SearchResult.scores_breakdown` carries bm25/vector/rrf/rerank components for debugging.
- Structured JSON query log: latency per stage, hit counts per index, top returned IDs.
- `LexicalIndex` now opens its connection with `check_same_thread=False` and wraps every read/write in a re-entrant lock so the async search pipeline can call it from worker threads safely.
- Added `tests/test_search.py` (8 tests) covering no-rerank top result, score-breakdown composition, rerank flipping order, per-call rerank toggle, empty-query guard, sync/async parity, missing-hydration skip, `k` truncation.

### Phase 4 — Fusion
- Added `reciprocal_rank_fusion` in `fusion.py`: pure function with `k=60` default, optional per-list weights, optional `top_n` truncation.
- Accepts either bare IDs or `(id, score)` tuples; ignores scores (uses rank only).
- Deduplicates within a single list (first occurrence wins) and across lists (additive scores).
- Zero-weight lists are skipped; deterministic tie-break on document ID keeps output stable.
- Added `tests/test_fusion.py` (14 tests) covering empty input, single-list passthrough, both-lists boost, weighting, tie-break determinism, formula correctness.

### Phase 3 — Vector Index
- Added `Embedder` (`embeddings.py`): lazy FastEmbed wrapper. Model loads on first embed call, not at construction. Batches to 32 by default.
- Added `VectorIndex` (`vector.py`): Qdrant-backed store with `connect(url, ...)` and `in_memory(...)` factories.
- Point IDs derived from chunk hex IDs (`int(chunk_id, 16) + 1`) so uint64 slots map 1:1 to chunks; point id 0 reserved for a meta sentinel recording the embedding model.
- `_ensure_collection` refuses to open a collection when stored model name or vector dim disagrees with the current embedder.
- API: `add`, `delete`, `delete_by_source`, `search` (cosine, filters out meta point), `count`.
- Added opt-in integration tests (`HYBRID_RUN_INTEGRATION=1`) for real `bge-small-en-v1.5` embedding dim and paraphrase similarity.
- Added `tests/conftest.py` with `FakeEmbedder` (SHA-256 hash → normalized vector) so unit tests run without network.
- Added `tests/test_vector.py` (10 tests) covering add/search/delete/count, meta exclusion, idempotent upsert, empty-query guard, model/dim mismatch refusal, 200-point batch.

### Phase 2 — Lexical Index
- Added `LexicalIndex` in `lexical.py`: SQLite FTS5 BM25 with `unicode61 remove_diacritics 2` tokenizer.
- Schema: regular `chunks` table (source of truth) + `chunks_fts` virtual table, kept in sync via single-transaction upserts.
- API: `add`, `delete`, `delete_by_source`, `search` (negates bm25 so higher = better), `get`, `get_many`, `chunk_ids_for_source`, `count`, `sources`.
- Query sanitizer strips punctuation and escapes embedded quotes so raw user input is safe to pass to FTS5 MATCH.
- WAL journaling + `synchronous=NORMAL` for faster bulk inserts.
- Added `tests/test_lexical.py` (10 tests) covering upserts, deletes, source deletes, ordered get_many, sanitization, 1000-chunk bulk insert, BM25 ranking, reopen-persists.

### Phase 1 — Chunking
- Added `Chunk` dataclass (frozen) with deterministic hash IDs (`make_chunk_id`).
- Added `TextChunker` — token-windowed via tiktoken `cl100k_base`, exact char offsets via prefix decode.
- Added `MarkdownChunker` — splits on headings, keeps fenced code blocks atomic, falls back to token chunker for oversized sections.
- Added `CodeChunker` — splits at top-level `def`/`class`/`function`/`func`/`fn`/`interface` starts with token fallback.
- Added `chunk_file` dispatcher picking a chunker from file extension.
- Added `tests/test_chunking.py` (17 tests) covering stable IDs, offset round-trip, token limits, fenced-code atomicity, dispatcher routing, reindex stability.

### Phase 0 — Project Setup
- Added `pyproject.toml` with core deps (qdrant-client, fastembed, fastapi, uvicorn, fastmcp, watchdog, pyyaml, tiktoken, pathspec, rich) and `dev`/`parsers` extras.
- Added `docker-compose.yml` running Qdrant on 6333/6334 with volume-mounted storage.
- Added `config.yaml` exposing every tuning knob listed in the plan.
- Added `Makefile` targets: `venv`, `install`, `up`, `down`, `logs`, `test`, `lint`, `clean`, `reset`.
- Added `src/hybrid_search/` package skeleton (module stubs for each phase) plus a real `config.load_config`.
- Added `tests/test_imports.py` smoke test covering every module plus the config loader.
