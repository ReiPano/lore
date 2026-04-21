# Hybrid Search

Local hybrid search (BM25 + vector) for MCP-compatible agents and plain HTTP clients. Long-term memory and retrieval over your own notes, code, and docs. See [PLAN.md](PLAN.md) for the design.

---

## Table of contents

- [What you get](#what-you-get)
- [Requirements](#requirements)
- [Install](#install)
- [First run](#first-run)
- [Daily usage](#daily-usage)
- [Managing projects](#managing-projects)
- [Querying](#querying)
- [Claude Code / Claude Desktop (MCP)](#claude-code--claude-desktop-mcp)
- [HTTP API](#http-api)
- [Configuration reference](#configuration-reference)
- [Exclusions](#exclusions)
- [Architecture](#architecture)
- [Evaluation](#evaluation)
- [Troubleshooting and pitfalls](#troubleshooting-and-pitfalls)
- [Uninstall](#uninstall)

---

## What you get

- **Hybrid retrieval** — SQLite FTS5 (BM25) + Qdrant (cosine) fused with Reciprocal Rank Fusion; optional cross-encoder reranker.
- **MCP server** — `search`, `index_path`, `list_sources`, `get_chunk` tools + `lore://sources` / `lore://source/{path}` resources.
- **HTTP API** — `POST /search`, `POST /index`, `GET /jobs/{id}`, `DELETE /documents`, `GET /health`.
- **Incremental indexing** — `watchdog`-backed watcher debounces editor saves, respects `.gitignore`, prunes common noise dirs.
- **Single CLI** — `lore init/up/down/restart/status/projects/tui/index/query/serve-api/serve-mcp/watch`.
- **User-scoped data** — everything under `~/.lore/` by default.

---

## Requirements

- macOS or Linux.
- Python 3.11 or 3.12.
- Docker (Docker Desktop on macOS). Qdrant runs as a container.
- About 300 MB free disk: ~130 MB for the embedding model cache, the rest for Qdrant + SQLite.

---

## Install

### 1. Clone and install

```bash
cd /path/where/you/keep/tools
git clone <repo-url> better-mem
cd better-mem
make install
```

`make install` does three things:

1. Creates `.venv/` with Python 3.12.
2. Installs the package + dev extras.
3. Runs `lore init`, which:
   - Creates `~/.lore/` and `~/.lore/qdrant/`.
   - Copies `config.yaml` to `~/.lore/config.yaml`.
   - Prints symlink + `$PATH` hints.

### 2. Put the CLI on your `$PATH`

The `init` command suggests a symlink. Example:

```bash
mkdir -p ~/.local/bin
ln -s /ABSOLUTE/PATH/TO/better-mem/.venv/bin/lore ~/.local/bin/lore
```

Ensure `~/.local/bin` is in `$PATH`. If not, for zsh:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify:

```bash
which lore
lore --help
```

### 3. Sanity-check tests (optional)

```bash
cd /path/to/better-mem
make test
```

---

## First run

```bash
lore up --watch          # starts Qdrant + (empty) watcher
lore status              # shows qdrant yes, watcher stopped, 0 chunks

lore projects add ~/notes --name notes
# ^ downloads bge-small-en-v1.5 (~130 MB, one time)
# ^ indexes all supported files under ~/notes

lore status              # chunk count > 0
lore query "something you wrote"
```

After your first `projects add`, restart services so the watcher picks up the new project:

```bash
lore restart
```

---

## Daily usage

```bash
lore up --watch          # morning
# ... work ...
lore restart             # after editing config, or if something feels stale
lore status              # any time
lore down                # evening
```

### Cheatsheet

| Command | What it does |
|---|---|
| `lore init` | Bootstrap `~/.lore/` and print PATH hints. |
| `lore up` | Start Qdrant (Docker). |
| `lore up --watch` | Start Qdrant and watcher daemon. |
| `lore down` | Stop watcher + Qdrant. |
| `lore restart` | Stop + restart watcher; start Qdrant if down. |
| `lore restart --full` | Also cycle Qdrant. |
| `lore restart --no-watch` | Stop the watcher only. |
| `lore status` | Table: Qdrant, watcher PID, chunk count, sources. |
| `lore projects list` | Show registered projects. |
| `lore projects add <path>` | Register a project and index it. |
| `lore projects add-all <path>` | Register every direct subfolder of `<path>` as its own project. |
| `lore projects remove <name>` | Unregister + drop chunks. |
| `lore projects reindex [<name>]` | Re-run indexing for one or all. |
| `lore index <path>` | One-shot index of a path (no registration). |
| `lore query <text>` | Ad-hoc query; prints JSON results. |
| `lore serve-api` | Run the FastAPI HTTP server. |
| `lore serve-mcp` | Run the FastMCP stdio server (used by Claude). |
| `lore watch [<paths>]` | Foreground watcher (blocks). |
| `lore tui` | Interactive menu. |
| `lore bench [args...]` | Run the token-usage benchmark (forwards args to `eval.token_bench.run_bench`). |

---

## Managing projects

Projects are named paths. Metadata lives in `~/.lore/projects.json`.

### Add

```bash
lore projects add ~/notes --name notes
lore projects add ~/code/app --name app
lore projects add ~/scratch --no-watch            # index once, do not watch
lore projects add ~/big-repo --skip-index         # register only; index later
```

Name is auto-derived from the path if you omit `--name`. Adding the same path twice updates in place.

### List / remove / reindex

```bash
lore projects list
lore projects remove app                          # also drops chunks
lore projects remove app --keep-index             # keep chunks
lore projects reindex                             # all
lore projects reindex notes                       # one
```

### Bulk add

Use `add-all` to register every direct subfolder of a path as its own project:

```bash
lore projects add-all ~/Projects/front-end
lore projects add-all ~/Projects/front-end --no-watch
lore projects add-all ~/Projects/front-end --skip-index   # register only
```

Each subfolder becomes a project whose name is the folder name. Hidden folders (`.foo`), built-in pruned dirs (`.git`, `node_modules`, …), and anything listed in `config.yaml:exclude_dirs` are skipped. Files directly under the target path are ignored — only immediate subdirectories are registered.

If you need finer control (custom names, selective watch flags), fall back to the loop form:

```bash
for dir in ~/Projects/front-end/*/; do
  lore projects add "$dir" --name "$(basename "$dir")"
done
```

### After changing which projects are watched

Bounce the daemon so it picks up new watched paths:

```bash
lore restart
```

---

## Querying

### Terminal

```bash
lore query "auth middleware" -k 5
lore query "auth middleware" --full               # full chunk text
lore query "auth middleware" --no-rerank          # skip cross-encoder
```

Output is one JSON object per line with `chunk_id`, `source_path`, `score`, `scores_breakdown` (bm25 / vector / rrf / rerank contributions), and `preview` or `text`.

### Interactive menu

```bash
lore tui
```

Menu-driven wrapper around the same commands.

---

## Claude Code / Claude Desktop (MCP)

### Register with Claude Code

```bash
claude mcp add lore -- \
  /ABSOLUTE/PATH/TO/better-mem/.venv/bin/lore \
  --config ~/.lore/config.yaml \
  serve-mcp
```

Order matters: `--config` goes **before** `serve-mcp`. Use absolute paths — Claude Code subprocess spawn does not expand `~` or honor your `$PATH`.

Verify:

```bash
claude mcp list | grep lore       # expect: ✓ Connected
```

Use Claude Code normally — tools `search` / `index_path` / `list_sources` / `get_chunk` appear automatically.

### Register with Claude Desktop

Copy `examples/claude_desktop_config.json`, fix the absolute paths, drop it into `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), and restart Claude Desktop.

```json
{
  "mcpServers": {
    "hybrid-search": {
      "command": "/ABSOLUTE/PATH/TO/better-mem/.venv/bin/lore",
      "args": ["--config", "/Users/<you>/.lore/config.yaml", "serve-mcp"]
    }
  }
}
```

---

## HTTP API

```bash
lore serve-api                 # default :8765
```

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"hybrid search","k":5}'
```

More examples in [examples/curl.sh](examples/curl.sh).

### Auth

Set `api.auth_token` in `~/.lore/config.yaml` to enable. Clients send `Authorization: Bearer <token>`.

---

## Configuration reference

Edit `~/.lore/config.yaml`. Apply changes with `lore restart`.

### Paths

| Field | Default | Meaning |
|---|---|---|
| `index_path` | `~/.lore/lexical.sqlite3` | SQLite file holding full chunk text + BM25 index. |
| `log_path` | `~/.lore/lore.log` | Rotating log file (2 MB × 5). |
| `qdrant_url` | `http://localhost:6333` | Where Qdrant is reachable. |
| `collection_name` | `hybrid_chunks` | Qdrant collection name. |

`~` and `$HOME` are expanded. All paths may be absolute or relative; relative paths are anchored to the cwd where you run the command.

### Chunking and retrieval

| Field | Default | Meaning |
|---|---|---|
| `chunk_size` | 500 | Tokens per chunk. |
| `chunk_overlap` | 50 | Tokens of overlap between neighbors. |
| `retrieval_k_per_index` | 50 | Candidates pulled from BM25 and vector before fusion. |
| `fusion_k` | 60 | RRF smoothing constant. |
| `bm25_weight` | 1.0 | RRF weight for BM25. |
| `vector_weight` | 1.0 | RRF weight for vector. |
| `rerank_top_n` | 20 | How many fused candidates the reranker scores. |
| `rerank_enabled` | true | Toggle cross-encoder reranking. |
| `default_result_k` | 10 | `k` when the caller omits it. |

### Models

| Field | Default | Meaning |
|---|---|---|
| `embedding_model` | `BAAI/bge-small-en-v1.5` | FastEmbed model. 384 dim, cached at `~/.cache/fastembed`. |
| `embedding_dim` | 384 | Must match the model. |
| `rerank_model` | `BAAI/bge-reranker-base` | FastEmbed cross-encoder. |

Changing the embedding model requires a fresh collection — Qdrant records the model name and refuses mismatches. Rebuild with `lore projects reindex` **after** wiping Qdrant data (`docker compose down -v`).

### Files and dirs

| Field | Default |
|---|---|
| `max_file_bytes` | 1 MB — files larger are skipped. |
| `supported_extensions` | Long whitelist including md/js/ts/py/java/go/rs and more. |
| `exclude_dirs` | Adds to the built-in pruned set (.git, node_modules, …). |
| `exclude_patterns` | Fnmatch globs run against filename, relative path, and full path. |

### API

| Field | Default |
|---|---|
| `api.host` | `127.0.0.1` |
| `api.port` | 8765 |
| `api.auth_token` | null |

### Config resolution order

When any command runs, the config path is resolved in this order:

1. `--config <path>` flag.
2. `$HYBRID_SEARCH_CONFIG`.
3. `~/.lore/config.yaml` (created by `init`).
4. `./config.yaml` in the current directory, walking up through parents.
5. `config.yaml` at the repo root (dev checkout fallback).

---

## Exclusions

Built-in pruned directories (always on):
`.git`, `.hg`, `.svn`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`, `.data`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`.

Default user-configured additions in `config.yaml` cover React, Angular, Next.js, Nuxt, SvelteKit, Java/Maven/Gradle, JetBrains/VS Code, macOS metadata, and common caches. Edit `exclude_dirs` / `exclude_patterns` to taste; see comments inline.

### Sensitive-file defaults

`config.yaml` ships two layers of protection:

- `exclude_patterns` adds common secret-bearing filenames (`.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `credentials.json`, `.npmrc`, `.pypirc`, `secret.yaml`).
- `exclude_content_patterns` is a regex list applied to the first 32 KB of every candidate file. Defaults match `sk-ant-…`, `sk-proj-…`, `AKIA…`, `ghp_…`, `gho_…`, `github_pat_…`, `xox[baprs]-…`, and inline PEM private keys. Any match makes the file skip indexing entirely and increments `files_skipped_sensitive` in the run stats.

Opt out by editing either list (or emptying it) in your user config. Opt in to new defaults after a future upgrade by running `lore init --force`.

Watcher and indexer share these rules, so ignored paths never enter the index even if you create them while the watcher is running.

---

## Architecture

```
 files → chunker → BM25 (SQLite FTS5)
                 → Vectors (Qdrant via FastEmbed)
                 → RRF fusion → (optional) cross-encoder rerank → results
                                                   ↑
                                 MCP server / HTTP API / CLI
```

- Chunk IDs are deterministic hashes of `(source_path, start_offset, text)`. Reindexing a file diffs new vs. existing and only reembeds the changed chunks.
- SQLite stores the full text, offsets, and metadata. Qdrant stores vectors plus a short preview. Full-text lookups always hit SQLite.
- The collection metadata point (id 0) records which embedding model built the index. Opening a collection with a different model raises.

---

## Evaluation

### Retrieval quality (hit@10 / MRR / NDCG@10)

```bash
.venv/bin/python -m eval.run_eval --fake-embedder          # smoke, no quality
.venv/bin/python -m eval.run_eval                           # real embedder
.venv/bin/python -m eval.run_eval --include-rerank          # also download reranker
```

Output lands in [eval/results.md](eval/results.md). Metrics: hit@10, MRR, NDCG@10. Extend [eval/queries.jsonl](eval/queries.jsonl) with your real queries; relevance is binary substring match against `must_contain`.

### Token-usage benchmark (plain vs MCP vs MCP+caveman)

```bash
.venv/bin/python -m eval.token_bench.run_bench --dry-run              # resolve paths only
.venv/bin/python -m eval.token_bench.run_bench --only plain --limit 1 # cheapest real run
.venv/bin/python -m eval.token_bench.run_bench                         # full matrix
lore bench --only plain --limit 1                             # equivalent via CLI
```

Drives `claude-agent-sdk` against the same task list under three configs (`plain`, `mcp`, `mcp+caveman`) and records input + output tokens per task. Needs `ANTHROPIC_API_KEY` set and Qdrant running for the MCP configs. Full instructions + task-file format in [eval/token_bench/README.md](eval/token_bench/README.md). Report lands in [eval/token_bench/results.md](eval/token_bench/results.md).

---

## Troubleshooting and pitfalls

### First-run gotchas

- **Model download blocks** — first `projects add` or `query` triggers a one-time ~130 MB download. Subsequent runs use the `~/.cache/fastembed` cache. If Ctrl-C'd mid-download you may re-download next time.
- **`no docker-compose.yml found`** — lifecycle now falls back to the repo root. If that still fails, make sure the repo still exists where you cloned it.
- **`FileNotFoundError: config not found`** — run `lore init` first, or pass `--config <path>`.

### MCP reports "✗ Failed to connect"

1. Run the server manually to see real errors:

   ```bash
   /ABS/PATH/.venv/bin/lore --config ~/.lore/config.yaml serve-mcp
   ```

   It should block silently (banner and logs are suppressed on stdio). Ctrl-C to exit.

2. Common root causes:
   - `--config` placed **after** `serve-mcp` in the MCP registration. Move it before.
   - Qdrant not running → `lore up`.
   - Absolute paths missing — Claude subprocess spawn does not expand `~` or read your shell `$PATH`.
   - First-time model download hung.

### Watcher

- **"no projects to watch; skipping watcher"** — you have no registered projects, or none are marked `watch: true`. Run `lore projects list` to confirm. After adding projects, run `lore restart`.
- **Daemon silently stops** — check `~/.lore/watcher.log`. Likely an unhandled exception or the parent `lore up` was killed.
- **macOS asks for permission to watch a folder** — accept. Watchdog uses FSEvents and needs per-folder access rights.

### Indexing

- **Progress bar stuck at some %** — a single big file may be embedding. `tail -f ~/.lore/lore.log` shows current file. Use `py-spy dump --pid $(pgrep -f reindex)` to see the exact stack if it truly hangs.
- **`project not found`** — you passed a name that does not match `lore projects list`. Use the name, not the path.
- **Index looks empty after restart** — Qdrant container storage lives under `~/.lore/qdrant/` (mounted via `HYBRID_QDRANT_DATA`). If you reset that directory, run `lore projects reindex` to rebuild.

### Data orphans

- Changing `index_path` or `embedding_model` in `config.yaml` strands old data. Either migrate files manually or run `make reset` + reindex.
- Qdrant keeps old vectors if you drop only the SQLite file. Hits without SQLite hydration are filtered out at query time (safe), but they waste space. `lore down && docker compose down -v && lore up --watch` is the nuclear cleanup.

### Config

- Do **not** comment out mandatory fields (`index_path`, `qdrant_url`, `collection_name`, `embedding_model`, `embedding_dim`, `chunk_size`, `chunk_overlap`) — load_config raises on missing keys.
- Paths support `~` and `$VARS`. Other fields do not.

### Uninstall

```bash
lore down
docker compose down -v                    # drop Qdrant volume
rm -rf ~/.lore
rm ~/.local/bin/lore             # if you symlinked
# then `rm -rf` the repo
```

Or, if the Makefile is available:

```bash
make reset         # stops services and wipes ~/.lore
make clean         # removes .venv
```

---

## License

MIT — see [LICENSE](LICENSE).
