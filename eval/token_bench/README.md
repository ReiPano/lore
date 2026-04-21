# Token-Usage Benchmark

Compares Claude token consumption across three configs on the same task list:

| config | MCP tools | Caveman skill |
|---|---|---|
| `plain` | ❌ | ❌ |
| `mcp` | ✅ (lore) | ❌ |
| `mcp+caveman` | ✅ (lore) | ✅ |

Each task runs once per enabled config. The runner sums `input_tokens` and `output_tokens` over every turn of the tool-using loop (including tool calls and tool results) and emits a markdown table.

## Requirements

- Authentication — **either** of:
  - Logged-in `claude` CLI (free if you have a Claude Pro/Max subscription; runs count against your Claude Code quota, no per-token cost). One-time `claude login`.
  - `ANTHROPIC_API_KEY` environment variable (console.anthropic.com → API Keys). Per-token billing against console credits.
  The runner auto-detects and prints which source is active on startup.
- `make install` to pull `claude-agent-sdk` (in the `dev` extras).
- Qdrant and the lore index running for the `mcp` configs:
  ```bash
  lore up
  lore projects add <path>
  ```
- Caveman skill installed under `~/.claude/plugins/cache/caveman/caveman/*/`. If missing, a short fallback stub is injected as a system prompt so the config still runs — a warning is logged.

## Running

```bash
# dry run: resolves paths, prints the plan, no API call
.venv/bin/python -m eval.token_bench.run_bench --dry-run

# smoke: single task, plain config only (cheapest real run)
.venv/bin/python -m eval.token_bench.run_bench --only plain --limit 1

# full: all three configs against all tasks
.venv/bin/python -m eval.token_bench.run_bench

# override model / task file / output
.venv/bin/python -m eval.token_bench.run_bench \
  --model claude-haiku-4-5 \
  --tasks path/to/my-tasks.jsonl \
  --output path/to/results.md
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` | `eval/token_bench/tasks.jsonl` | Task file (one JSON per line). |
| `--output` | `eval/token_bench/results.md` | Markdown report destination. |
| `--model` | `claude-sonnet-4-5` | Model ID passed to the SDK. |
| `--max-turns` | 12 | Per-task turn budget. |
| `--only` | — | Repeatable; restrict to `plain` / `mcp` / `mcp+caveman`. |
| `--limit` | — | Cap number of tasks. |
| `--cli-binary` | `.venv/bin/lore` | Binary the SDK spawns for the MCP server. |
| `--hybrid-config` | auto-resolved | Config file passed to the MCP server. |
| `--dry-run` | off | Print plan + exit. |
| `-v` / `--verbose` | off | INFO-level logs from the runner. |

## Task file format

JSONL, one task per line:

```json
{"id": "project-overview", "prompt": "5-bullet overview of indexed projects.", "notes": "mcp lookup"}
```

Add your own tasks by appending to `tasks.jsonl` or pointing `--tasks` at a custom file.

## Output

`results.md` contains a totals table plus a per-task breakdown showing total tokens per config per task. Commit new runs so regressions are visible in diffs.

## Cost note

- On **Claude Pro/Max subscription** (via `claude login`): runs consume from your Claude Code quota. No per-token dollar cost, but high-volume runs can hit daily caps.
- On **API key**: single-digit cents for the default 8×3 Sonnet run; tool-heavy tasks climb fast with high `--max-turns`.
- Start with `--only plain --limit 1` while iterating on tasks.

## Caveats

- The `mcp` configs need the live MCP server; if Qdrant is down the MCP tool call fails and the task is logged as errored (token totals may still include partial usage).
- Caveman is injected both as a local plugin (so its hooks wire up if the SDK supports them) and as a system prompt (so its style instructions fire regardless). This double-anchor keeps the benchmark robust against plugin-layout changes.
- `max_turns=12` is a ceiling. If a task consistently hits it, raise the flag to distinguish "slow config" from "runaway tool loop".
- Only input + output tokens are measured. Cache tokens, latency, and cost are explicitly deferred.
