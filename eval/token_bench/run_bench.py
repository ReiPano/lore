"""Token-usage benchmark across three agent configurations.

Drives ``claude-agent-sdk`` against the same task set under:

* ``plain``        — no MCP servers, no skill.
* ``mcp``          — hybrid-search MCP server registered.
* ``mcp+caveman``  — ``mcp`` plus the caveman plugin loaded as an SDK plugin.

Each task runs once per enabled config. Input and output tokens (summed across
every turn of the tool-using loop) are recorded. A markdown table is written to
``--output`` so runs can be diffed over time.

Cost warning: every non-``--dry-run`` invocation sends real requests to the
Anthropic API. Keep task lists short while iterating on the harness.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = ROOT / "tasks.jsonl"
DEFAULT_RESULTS = ROOT / "results.md"

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TURNS = 12

ALL_CONFIGS = ("plain", "mcp", "mcp+caveman")

CAVEMAN_PLUGIN_GLOB = "~/.claude/plugins/cache/caveman/caveman/*"
CAVEMAN_SKILL_GLOBS = (
    "~/.claude/plugins/cache/caveman/caveman/*/caveman/SKILL.md",
    "~/.claude/plugins/cache/caveman/caveman/*/skills/caveman/SKILL.md",
)

CAVEMAN_FALLBACK = (
    "Caveman mode: drop articles (a/an/the), filler (just/really/basically), "
    "and pleasantries. Fragments OK. Short synonyms. Keep all technical "
    "substance exact. Code blocks unchanged. Errors quoted exact."
)


@dataclass(slots=True)
class Task:
    id: str
    prompt: str
    notes: str = ""


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    duration_ms: int = 0
    errored: bool = False

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.turns += other.turns
        self.duration_ms += other.duration_ms
        self.errored = self.errored or other.errored


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        tasks.append(
            Task(
                id=str(obj["id"]),
                prompt=str(obj["prompt"]),
                notes=str(obj.get("notes", "")),
            )
        )
    return tasks


# ---- caveman discovery ----------------------------------------------------


def _expand(pattern: str) -> str:
    return os.path.expanduser(pattern)


def find_caveman_plugin() -> Path | None:
    for candidate in sorted(glob.glob(_expand(CAVEMAN_PLUGIN_GLOB))):
        path = Path(candidate)
        if path.is_dir():
            return path
    return None


def find_caveman_skill() -> Path | None:
    for pattern in CAVEMAN_SKILL_GLOBS:
        matches = sorted(glob.glob(_expand(pattern)))
        if matches:
            return Path(matches[0])
    return None


def caveman_skill_text() -> str:
    skill = find_caveman_skill()
    if skill is None:
        log.warning("caveman SKILL.md not found; using fallback stub")
        return CAVEMAN_FALLBACK
    raw = skill.read_text(encoding="utf-8")
    # Strip a leading YAML frontmatter block if present.
    body = re.sub(r"^---\n.*?\n---\n", "", raw, count=1, flags=re.DOTALL)
    return body.strip()


# ---- SDK wiring ------------------------------------------------------------


def make_options(
    *,
    kind: str,
    model: str,
    max_turns: int,
    cli_binary: Path,
    hybrid_config: Path,
    caveman_plugin: Path | None,
    caveman_text: str,
):
    from claude_agent_sdk import ClaudeAgentOptions

    mcp_servers: dict[str, dict[str, Any]] = {}
    allowed_tools: list[str] = []
    plugins: list[dict[str, str]] = []
    system_prompt: str | None = None

    if kind in {"mcp", "mcp+caveman"}:
        mcp_servers["lore"] = {
            "type": "stdio",
            "command": str(cli_binary),
            "args": ["--config", str(hybrid_config), "serve-mcp"],
        }
        allowed_tools = [
            "mcp__lore__search",
            "mcp__lore__index_path",
            "mcp__lore__list_sources",
            "mcp__lore__get_chunk",
        ]
    if kind == "mcp+caveman":
        if caveman_plugin is not None:
            plugins.append({"type": "local", "path": str(caveman_plugin)})
        # Always inject the skill text as a system prompt so caveman's style
        # instructions fire even if the plugin manifest changes structure.
        system_prompt = caveman_text

    return ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        mcp_servers=mcp_servers if mcp_servers else {},
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        plugins=plugins,
        permission_mode="bypassPermissions",
    )


async def run_task(options, prompt: str) -> Usage:
    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk.types import AssistantMessage, ResultMessage

    usage = Usage()
    started = time.perf_counter()

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                if message.usage:
                    usage.input_tokens = int(message.usage.get("input_tokens", 0))
                    usage.output_tokens = int(message.usage.get("output_tokens", 0))
                usage.turns = int(message.num_turns or 0)
                usage.duration_ms = int(message.duration_ms or 0)
                usage.errored = bool(message.is_error)
                break
            if isinstance(message, AssistantMessage) and message.usage and usage.input_tokens == 0:
                # Fallback: accumulate across assistant turns if ResultMessage lacks usage.
                usage.input_tokens += int(message.usage.get("input_tokens", 0))
                usage.output_tokens += int(message.usage.get("output_tokens", 0))

    if usage.duration_ms == 0:
        usage.duration_ms = int((time.perf_counter() - started) * 1000)
    return usage


# ---- orchestration --------------------------------------------------------


@dataclass(slots=True)
class BenchResult:
    config: str
    per_task: dict[str, Usage] = field(default_factory=dict)

    def total(self) -> Usage:
        agg = Usage()
        for u in self.per_task.values():
            agg.merge(u)
        return agg


async def run_bench(
    *,
    tasks: list[Task],
    configs: list[str],
    model: str,
    max_turns: int,
    cli_binary: Path,
    hybrid_config: Path,
) -> list[BenchResult]:
    caveman_plugin = find_caveman_plugin()
    caveman_text = caveman_skill_text()
    results: list[BenchResult] = []
    for kind in configs:
        bench_result = BenchResult(config=kind)
        options = make_options(
            kind=kind,
            model=model,
            max_turns=max_turns,
            cli_binary=cli_binary,
            hybrid_config=hybrid_config,
            caveman_plugin=caveman_plugin,
            caveman_text=caveman_text,
        )
        for task in tasks:
            log.info("[%s] running task %s", kind, task.id)
            try:
                usage = await run_task(options, task.prompt)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] task %s failed", kind, task.id)
                usage = Usage(errored=True)
                usage.duration_ms = 0
                _ = exc
            bench_result.per_task[task.id] = usage
        results.append(bench_result)
    return results


# ---- reporting -------------------------------------------------------------


def format_markdown(
    results: list[BenchResult],
    *,
    model: str,
    tasks: list[Task],
) -> str:
    lines = [
        "# Token-Usage Benchmark",
        "",
        f"- model: `{model}`",
        f"- tasks: {len(tasks)}",
        f"- configs: {', '.join(r.config for r in results)}",
        "",
        "## Totals",
        "",
        "| config | input tokens | output tokens | total | turns | duration ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r.total()
        lines.append(
            f"| {r.config} | {t.input_tokens:,} | {t.output_tokens:,} | "
            f"{t.total:,} | {t.turns} | {t.duration_ms:,} |"
        )

    lines.append("")
    lines.append("## Per-task (total tokens)")
    lines.append("")
    header = "| task | " + " | ".join(r.config for r in results) + " |"
    sep = "|---" * (1 + len(results)) + "|"
    lines.append(header)
    lines.append(sep)
    for task in tasks:
        cells = [f"`{task.id}`"]
        for r in results:
            u = r.per_task.get(task.id)
            cell = "—" if u is None else (f"**err** {u.total:,}" if u.errored else f"{u.total:,}")
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---- CLI ------------------------------------------------------------------


def _resolve_cli_binary(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    # Prefer the venv binary that lives next to the runner's interpreter.
    venv_bin = Path(sys.executable).resolve().parent / "hybrid-search"
    if venv_bin.exists():
        return venv_bin
    # Fallback to `hybrid-search` on PATH.
    import shutil

    located = shutil.which("hybrid-search")
    if located:
        return Path(located)
    raise FileNotFoundError(
        "cannot locate hybrid-search binary; pass --cli-binary"
    )


def _resolve_hybrid_config(explicit: Path | None) -> Path:
    from hybrid_search.config import resolve_config_path

    if explicit is not None:
        return resolve_config_path(explicit)
    return resolve_config_path(None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument(
        "--only",
        action="append",
        choices=list(ALL_CONFIGS),
        help="Run only the given config(s). Repeatable.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap the number of tasks.")
    p.add_argument("--cli-binary", type=Path, default=None)
    p.add_argument("--hybrid-config", type=Path, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths + print the plan without calling the API.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s - %(message)s")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    configs = list(args.only) if args.only else list(ALL_CONFIGS)

    cli_binary = _resolve_cli_binary(args.cli_binary)
    hybrid_config = _resolve_hybrid_config(args.hybrid_config)
    caveman_plugin = find_caveman_plugin()

    auth_source = (
        "ANTHROPIC_API_KEY"
        if os.environ.get("ANTHROPIC_API_KEY")
        else ("claude CLI (subscription)" if _has_auth() else "(none detected)")
    )
    print(f"model:          {args.model}")
    print(f"cli binary:     {cli_binary}")
    print(f"hybrid config:  {hybrid_config}")
    print(f"caveman plugin: {caveman_plugin or '(not found; fallback stub)'}")
    print(f"tasks:          {len(tasks)} ({args.tasks})")
    print(f"configs:        {', '.join(configs)}")
    print(f"auth:           {auth_source}")
    print()

    if args.dry_run:
        print("dry run — no API calls made")
        return 0

    if not _has_auth():
        print(
            "error: no usable auth. Either run `claude login` (uses your Claude "
            "Code subscription) or export ANTHROPIC_API_KEY (billed per token).",
            file=sys.stderr,
        )
        return 2

    if any(c in {"mcp", "mcp+caveman"} for c in configs):
        if not _qdrant_reachable(hybrid_config):
            print(
                "warning: Qdrant does not appear reachable; MCP configs will likely fail",
                file=sys.stderr,
            )

    results = asyncio.run(
        run_bench(
            tasks=tasks,
            configs=configs,
            model=args.model,
            max_turns=args.max_turns,
            cli_binary=cli_binary,
            hybrid_config=hybrid_config,
        )
    )

    report = format_markdown(results, model=args.model, tasks=tasks)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    return 0


def _has_auth() -> bool:
    """Accept either an Anthropic API key or a logged-in Claude Code CLI.

    The SDK spawns the ``claude`` CLI, so if it is installed and the user has
    run ``claude login`` (for a Pro/Max subscription), no API key is needed.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    import shutil

    return shutil.which("claude") is not None


def _qdrant_reachable(hybrid_config: Path) -> bool:
    import socket
    from urllib.parse import urlparse

    import yaml

    try:
        raw = yaml.safe_load(hybrid_config.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return True  # best effort; don't block
    url = urlparse(raw.get("qdrant_url", "http://localhost:6333"))
    host = url.hostname or "localhost"
    port = url.port or 6333
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


# Small convenience re-export for tests.
def iter_configs() -> Iterable[str]:
    return iter(ALL_CONFIGS)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
