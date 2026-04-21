"""Tests for the token-usage benchmark harness.

Exercises only code that does not touch the Anthropic API: task parsing,
caveman skill resolution, option assembly, and markdown rendering.
"""

from __future__ import annotations

from pathlib import Path

from eval.token_bench.run_bench import (
    ALL_CONFIGS,
    BenchResult,
    CAVEMAN_FALLBACK,
    Task,
    Usage,
    caveman_skill_text,
    find_caveman_skill,
    format_markdown,
    load_tasks,
    make_options,
)

CAVEMAN_MODULE = "eval.token_bench.run_bench"


def test_load_tasks_shipped_file_is_usable() -> None:
    path = Path(__file__).resolve().parents[1] / "eval" / "token_bench" / "tasks.jsonl"
    tasks = load_tasks(path)
    assert len(tasks) >= 6
    ids = {t.id for t in tasks}
    assert "project-overview" in ids
    assert all(isinstance(t.prompt, str) and t.prompt.strip() for t in tasks)


def test_load_tasks_ignores_blank_and_comment_lines(tmp_path: Path) -> None:
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        "\n"
        '{"id": "a", "prompt": "one"}\n'
        "# a comment\n"
        "\n"
        '{"id": "b", "prompt": "two", "notes": "hi"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["a", "b"]
    assert tasks[1].notes == "hi"


def test_caveman_skill_text_fallback(monkeypatch) -> None:
    monkeypatch.setattr(f"{CAVEMAN_MODULE}.find_caveman_skill", lambda: None)
    text = caveman_skill_text()
    assert text == CAVEMAN_FALLBACK


def test_caveman_skill_text_strips_frontmatter(monkeypatch, tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\n"
        "name: caveman\n"
        "version: 1\n"
        "---\n"
        "Caveman body here.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(f"{CAVEMAN_MODULE}.find_caveman_skill", lambda: skill)
    text = caveman_skill_text()
    assert "Caveman body here." in text
    assert "---" not in text


def test_make_options_plain_has_no_servers(tmp_path: Path) -> None:
    opts = make_options(
        kind="plain",
        model="claude-test-model",
        max_turns=5,
        cli_binary=tmp_path / "lore",
        hybrid_config=tmp_path / "config.yaml",
        caveman_plugin=None,
        caveman_text=CAVEMAN_FALLBACK,
    )
    assert opts.model == "claude-test-model"
    assert opts.mcp_servers == {}
    assert opts.allowed_tools == []
    assert opts.plugins == []
    assert opts.system_prompt is None


def test_make_options_mcp_populates_server(tmp_path: Path) -> None:
    opts = make_options(
        kind="mcp",
        model="claude-test-model",
        max_turns=5,
        cli_binary=tmp_path / "lore",
        hybrid_config=tmp_path / "config.yaml",
        caveman_plugin=None,
        caveman_text=CAVEMAN_FALLBACK,
    )
    assert "lore" in opts.mcp_servers
    server = opts.mcp_servers["lore"]
    assert server["command"].endswith("lore")
    assert "serve-mcp" in server["args"]
    assert "mcp__lore__search" in opts.allowed_tools
    assert opts.system_prompt is None


def test_make_options_caveman_adds_plugin_and_system(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    opts = make_options(
        kind="mcp+caveman",
        model="claude-test-model",
        max_turns=5,
        cli_binary=tmp_path / "lore",
        hybrid_config=tmp_path / "config.yaml",
        caveman_plugin=plugin_dir,
        caveman_text="CAVEMAN TEXT",
    )
    assert opts.system_prompt == "CAVEMAN TEXT"
    assert any(p["path"] == str(plugin_dir) for p in opts.plugins)
    assert opts.mcp_servers and "lore" in opts.mcp_servers


def test_format_markdown_renders_totals_and_per_task() -> None:
    tasks = [Task(id="t1", prompt="p"), Task(id="t2", prompt="q")]
    plain = BenchResult(config="plain")
    plain.per_task = {
        "t1": Usage(input_tokens=1000, output_tokens=200, turns=1, duration_ms=500),
        "t2": Usage(input_tokens=500, output_tokens=100, turns=1, duration_ms=300),
    }
    mcp = BenchResult(config="mcp")
    mcp.per_task = {
        "t1": Usage(input_tokens=800, output_tokens=150, turns=2, duration_ms=600),
        "t2": Usage(input_tokens=400, output_tokens=80, turns=2, duration_ms=400, errored=True),
    }
    out = format_markdown([plain, mcp], model="claude-test-model", tasks=tasks)
    assert "claude-test-model" in out
    # Totals: plain = 1000+200 + 500+100 = 1800; mcp = 800+150 + 400+80 = 1430
    assert "1,800" in out
    assert "1,430" in out
    # Per-task row has both config columns.
    assert "`t1`" in out and "`t2`" in out
    # Errored task shows the err marker.
    assert "**err**" in out


def test_all_configs_constant_is_stable() -> None:
    assert ALL_CONFIGS == ("plain", "mcp", "mcp+caveman")


def test_find_caveman_skill_is_callable() -> None:
    result = find_caveman_skill()
    assert result is None or result.name == "SKILL.md"
