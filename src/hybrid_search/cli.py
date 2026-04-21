"""Command-line entry points.

Subcommands
-----------

Lifecycle:
    hybrid-search up                 Start Qdrant (optionally spawn watcher).
    hybrid-search down               Stop Qdrant + watcher.
    hybrid-search status             Show running services + index stats.

Indexing:
    hybrid-search index <path>       Index files/directories.
    hybrid-search query <text>       One-shot hybrid query.
    hybrid-search watch              Run the file watcher (foreground).

Projects:
    hybrid-search projects list
    hybrid-search projects add <path> [--name N] [--no-watch]
    hybrid-search projects remove <name_or_path>
    hybrid-search projects reindex [name_or_path]

Servers:
    hybrid-search serve-api          FastAPI HTTP server.
    hybrid-search serve-mcp          FastMCP stdio server.

Interactive:
    hybrid-search tui                Interactive menu-driven mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import USER_CONFIG_PATH, USER_DATA_DIR, Config, load_config
from .embeddings import Embedder
from .indexer import Indexer, IndexStats
from .lexical import LexicalIndex
from .lifecycle import (
    qdrant_down,
    qdrant_status,
    qdrant_up,
    watcher_running,
    watcher_start,
    watcher_stop,
)
from .logging_setup import configure_logging
from .projects import ProjectStore
from .rerank import Reranker
from .search import HybridSearch
from .vector import VectorIndex

log = logging.getLogger(__name__)
console = Console()


# ---- wiring helpers -------------------------------------------------------


def _build_stack(cfg: Config) -> tuple[LexicalIndex, VectorIndex, Embedder]:
    embedder = Embedder(model_name=cfg.embedding_model)
    lex = LexicalIndex(cfg.index_path)
    vec = VectorIndex.connect(
        cfg.qdrant_url,
        cfg.collection_name,
        embedder,
        dim=cfg.embedding_dim,
    )
    return lex, vec, embedder


def _build_indexer(cfg: Config, lex: LexicalIndex, vec: VectorIndex) -> Indexer:
    return Indexer(
        lex,
        vec,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        supported_extensions=cfg.supported_extensions or None,
        max_file_bytes=cfg.max_file_bytes,
        show_progress=True,
        exclude_dirs=cfg.exclude_dirs or None,
        exclude_patterns=cfg.exclude_patterns or None,
        exclude_content_patterns=cfg.exclude_content_patterns or None,
    )


def _build_search(cfg: Config, lex: LexicalIndex, vec: VectorIndex) -> HybridSearch:
    reranker = (
        Reranker(model_name=cfg.rerank_model)
        if cfg.rerank_enabled and cfg.rerank_model
        else None
    )
    return HybridSearch(
        lex,
        vec,
        reranker=reranker,
        retrieval_k_per_index=cfg.retrieval_k_per_index,
        fusion_k=cfg.fusion_k,
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
        rerank_top_n=cfg.rerank_top_n,
        default_k=cfg.default_result_k,
        rerank_enabled=cfg.rerank_enabled,
    )


def _data_dir(cfg: Config) -> Path:
    parent = Path(cfg.index_path).parent
    if not parent or str(parent) in ("", "."):
        parent = Path.cwd() / ".data"
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _project_store(cfg: Config) -> ProjectStore:
    return ProjectStore(_data_dir(cfg) / "projects.json")


def _merged(a: IndexStats, b: IndexStats) -> IndexStats:
    a.merge(b)
    return a


# ---- lifecycle commands ---------------------------------------------------


def _qdrant_data_dir(cfg: Config) -> Path:
    return _data_dir(cfg) / "qdrant"


def _cmd_init(args: argparse.Namespace) -> int:
    """Bootstrap ~/.lore with a config, data dirs, and PATH hints.

    Runs without needing an existing config — argparse short-circuits the
    usual ``load_config`` call for this command.
    """

    import os
    import shutil

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (USER_DATA_DIR / "qdrant").mkdir(exist_ok=True)

    force = bool(getattr(args, "force", False))
    template = Path(__file__).resolve().parents[2] / "config.yaml"
    if not template.exists():
        console.print(f"[red]template config missing: {template}[/red]")
        return 1

    if USER_CONFIG_PATH.exists() and not force:
        console.print(f"[yellow]config already exists at {USER_CONFIG_PATH}[/yellow]")
        console.print("use --force to overwrite")
    else:
        shutil.copy2(template, USER_CONFIG_PATH)
        console.print(f"[green]wrote config[/green] {USER_CONFIG_PATH}")

    venv_bin = Path(sys.executable).resolve().parent
    cli_bin = venv_bin / "hybrid-search"

    console.rule("[bold cyan]install summary[/bold cyan]")
    console.print(f"data dir:   {USER_DATA_DIR}")
    console.print(f"config:     {USER_CONFIG_PATH}")
    console.print(f"CLI binary: {cli_bin}")
    console.print()

    # Symlink recommendation for PATH.
    target = Path.home() / ".local" / "bin" / "hybrid-search"
    console.rule("[bold cyan]add to PATH[/bold cyan]")
    if target.exists() or target.is_symlink():
        console.print(f"[green]symlink already exists:[/green] {target}")
    else:
        console.print(f"mkdir -p {target.parent}")
        console.print(f"ln -s {cli_bin} {target}")
    shell = os.environ.get("SHELL", "/bin/zsh")
    rc = "~/.zshrc" if "zsh" in shell else "~/.bashrc"
    console.print()
    console.print(f"ensure {target.parent} is in $PATH; if not, add to {rc}:")
    console.print(f'  export PATH="{target.parent}:$PATH"')
    console.print(f"then: source {rc}")

    console.rule("[bold cyan]next steps[/bold cyan]")
    console.print("1. hybrid-search up --watch")
    console.print("2. hybrid-search projects add <path>")
    console.print("3. hybrid-search status")
    console.print("4. register MCP with Claude Code:")
    console.print(
        f"   claude mcp add hybrid-search -- {cli_bin} "
        f"--config {USER_CONFIG_PATH} serve-mcp"
    )
    return 0


def _cmd_up(args: argparse.Namespace, cfg: Config) -> int:
    ok, msg = qdrant_up(qdrant_data=_qdrant_data_dir(cfg))
    if not ok:
        console.print(f"[red]qdrant up failed:[/red] {msg}")
        return 1
    console.print("[green]qdrant running[/green]")

    if args.watch:
        store = _project_store(cfg)
        paths = store.watched_paths() or [str(p) for p in cfg.watch_paths]
        if not paths:
            console.print("[yellow]no projects to watch; skipping watcher[/yellow]")
            return 0
        try:
            handle = watcher_start(
                data_dir=_data_dir(cfg),
                config_path=_resolve_config_for_watcher(args),
                paths=paths,
            )
        except RuntimeError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            return 0
        console.print(f"[green]watcher started (pid {handle.pid})[/green]")
    return 0


def _cmd_down(args: argparse.Namespace, cfg: Config) -> int:
    stopped = watcher_stop(_data_dir(cfg))
    if stopped:
        console.print("[green]watcher stopped[/green]")
    ok, msg = qdrant_down(qdrant_data=_qdrant_data_dir(cfg))
    console.print(f"[{'green' if ok else 'red'}]{msg}[/]")
    return 0 if ok else 1


def _cmd_restart(args: argparse.Namespace, cfg: Config) -> int:
    # Always bounce the watcher. Qdrant only cycles if --full was passed.
    stopped = watcher_stop(_data_dir(cfg))
    if stopped:
        console.print("[yellow]watcher stopped[/yellow]")

    if args.full:
        ok, msg = qdrant_down(qdrant_data=_qdrant_data_dir(cfg))
        console.print(f"[{'yellow' if ok else 'red'}]{msg}[/]")
        ok, msg = qdrant_up(qdrant_data=_qdrant_data_dir(cfg))
        if not ok:
            console.print(f"[red]qdrant up failed:[/red] {msg}")
            return 1
        console.print("[green]qdrant running[/green]")
    else:
        status = qdrant_status(qdrant_data=_qdrant_data_dir(cfg))
        if status.get("running") != "yes":
            ok, msg = qdrant_up(qdrant_data=_qdrant_data_dir(cfg))
            if not ok:
                console.print(f"[red]qdrant up failed:[/red] {msg}")
                return 1
            console.print("[green]qdrant running[/green]")

    watch_flag = not args.no_watch
    if not watch_flag:
        return 0
    store = _project_store(cfg)
    paths = store.watched_paths() or [str(p) for p in cfg.watch_paths]
    if not paths:
        console.print("[yellow]no projects to watch; skipping watcher[/yellow]")
        return 0
    try:
        handle = watcher_start(
            data_dir=_data_dir(cfg),
            config_path=_resolve_config_for_watcher(args),
            paths=paths,
        )
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return 0
    console.print(f"[green]watcher started (pid {handle.pid})[/green]")
    return 0


def _resolve_config_for_watcher(args: argparse.Namespace) -> Path:
    from .config import resolve_config_path

    return resolve_config_path(getattr(args, "config", None))


def _cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    qdrant = qdrant_status()
    watcher_pid = watcher_running(_data_dir(cfg))
    projects = _project_store(cfg).load()

    # Try to read index counts (requires Qdrant up). Ignore failures.
    chunk_count = None
    sources_count = None
    try:
        lex, vec, _ = _build_stack(cfg)
        try:
            chunk_count = lex.count()
            sources_count = len(lex.sources())
        finally:
            lex.close()
            vec.close()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]index stats unavailable: {exc}[/yellow]")

    table = Table(title="hybrid-search status", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("qdrant", qdrant["running"])
    table.add_row("qdrant detail", qdrant.get("detail", "")[:120])
    table.add_row("watcher", f"running (pid {watcher_pid})" if watcher_pid else "stopped")
    table.add_row("projects", str(len(projects)))
    if chunk_count is not None:
        table.add_row("chunks", str(chunk_count))
        table.add_row("sources", str(sources_count))
    console.print(table)
    return 0


# ---- project commands -----------------------------------------------------


def _cmd_projects_list(args: argparse.Namespace, cfg: Config) -> int:
    projects = _project_store(cfg).load()
    if not projects:
        console.print("[yellow]no projects registered[/yellow]")
        return 0
    table = Table(title="projects")
    table.add_column("name", style="cyan")
    table.add_column("path")
    table.add_column("watch")
    for p in projects:
        table.add_row(p.name, p.path, "yes" if p.watch else "no")
    console.print(table)
    return 0


def _cmd_projects_add(args: argparse.Namespace, cfg: Config) -> int:
    verbose = bool(getattr(args, "verbose", False))
    store = _project_store(cfg)
    project = store.add(args.path, name=args.name, watch=not args.no_watch)
    console.print(
        f"[green]added[/green] [cyan]{project.name}[/cyan]"
        + (" (watched)" if project.watch else " (manual)")
    )
    if not args.skip_index:
        lex, vec, _ = _build_stack(cfg)
        try:
            indexer = _build_indexer(cfg, lex, vec)
            indexer.progress_label = project.name
            stats = indexer.index_path(project.path)
            _print_index_summary(project.name, stats, verbose=verbose)
        finally:
            lex.close()
            vec.close()
    # If watcher is running, bounce so it picks up the new path.
    if watcher_running(_data_dir(cfg)) is not None:
        console.print("[yellow]watcher running; `hybrid-search restart` to pick up new path[/yellow]")
    return 0


def _print_index_summary(name: str, stats, *, verbose: bool) -> None:
    if verbose:
        console.print_json(json.dumps(asdict(stats), default=str))
        if stats.errors:
            console.print("[red]errors:[/red]")
            for path, msg in stats.errors[:20]:
                console.print(f"  {path}: {msg}")
            if len(stats.errors) > 20:
                console.print(f"  … {len(stats.errors) - 20} more")
        return
    extras = []
    if stats.files_skipped_unsupported:
        extras.append(f"{stats.files_skipped_unsupported} skipped (ext)")
    if stats.files_skipped_too_large:
        extras.append(f"{stats.files_skipped_too_large} skipped (size)")
    skipped_sensitive = getattr(stats, "files_skipped_sensitive", 0)
    if skipped_sensitive:
        extras.append(f"{skipped_sensitive} skipped (sensitive)")
    if stats.files_skipped_unchanged:
        extras.append(f"{stats.files_skipped_unchanged} unchanged")
    if stats.errors:
        extras.append(f"[red]{len(stats.errors)} errors[/red]")
    tail = f" ({', '.join(extras)})" if extras else ""
    console.print(
        f"  [cyan]{name}[/cyan]: +{stats.chunks_added} chunks "
        f"in {stats.files_indexed} files"
        f" -{stats.chunks_removed} removed" + tail
    )
    if stats.errors:
        # Always show the first couple of error messages so silent failures
        # never slip past the default summary. Use -v for the full dump.
        for path, msg in stats.errors[:3]:
            console.print(f"    [red]!{msg}[/red]  {path}")
        if len(stats.errors) > 3:
            console.print(
                f"    [red](… {len(stats.errors) - 3} more; use -v to see all)[/red]"
            )


def _cmd_projects_add_all(args: argparse.Namespace, cfg: Config) -> int:
    verbose = bool(getattr(args, "verbose", False))
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        console.print(f"[red]not a directory: {root}[/red]")
        return 1

    from .indexer import PRUNED_DIRS

    children = sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name not in PRUNED_DIRS
        and p.name not in set(cfg.exclude_dirs)
    )
    if not children:
        console.print(f"[yellow]no eligible subfolders under {root}[/yellow]")
        return 0

    store = _project_store(cfg)
    added: list = []
    for child in children:
        project = store.add(child, name=child.name, watch=not args.no_watch)
        added.append(project)

    console.print(
        f"[green]registered[/green] {len(added)} projects "
        + ("(watched)" if not args.no_watch else "(manual)")
    )
    if verbose:
        for p in added:
            console.print(f"  [cyan]{p.name}[/cyan] -> {p.path}")

    if args.skip_index:
        if watcher_running(_data_dir(cfg)) is not None:
            console.print(
                "[yellow]watcher running; `hybrid-search restart` to pick up new paths[/yellow]"
            )
        return 0

    lex, vec, _ = _build_stack(cfg)
    try:
        indexer = _build_indexer(cfg, lex, vec)
        from .indexer import IndexStats

        total = IndexStats()
        for project in added:
            indexer.progress_label = project.name
            stats = indexer.index_path(project.path)
            total = _merged(total, stats)
            _print_index_summary(project.name, stats, verbose=verbose)
        if verbose:
            console.print_json(json.dumps(asdict(total), default=str))
        else:
            console.print(
                f"[bold green]done[/bold green]: +{total.chunks_added} chunks "
                f"across {total.files_indexed} files in {len(added)} projects"
            )
    finally:
        lex.close()
        vec.close()

    if watcher_running(_data_dir(cfg)) is not None:
        console.print(
            "[yellow]watcher running; `hybrid-search restart` to pick up new paths[/yellow]"
        )
    return 0


def _cmd_projects_remove(args: argparse.Namespace, cfg: Config) -> int:
    store = _project_store(cfg)
    project = store.find(args.target)
    if project is None:
        console.print(f"[red]no project matching {args.target!r}[/red]")
        return 1
    if not args.keep_index:
        lex, vec, _ = _build_stack(cfg)
        try:
            indexer = _build_indexer(cfg, lex, vec)
            stats = indexer.remove_path(project.path)
            console.print_json(json.dumps(asdict(stats), default=str))
        finally:
            lex.close()
            vec.close()
    store.remove(project.name)
    console.print(f"[green]removed[/green] {project.name}")
    return 0


def _cmd_projects_reindex(args: argparse.Namespace, cfg: Config) -> int:
    verbose = bool(getattr(args, "verbose", False))
    store = _project_store(cfg)
    targets = store.load() if not args.target else [store.find(args.target)]
    if any(t is None for t in targets):
        console.print(f"[red]no project matching {args.target!r}[/red]")
        return 1
    lex, vec, _ = _build_stack(cfg)
    try:
        indexer = _build_indexer(cfg, lex, vec)
        total = IndexStats()
        for project in targets:
            indexer.progress_label = project.name
            stats = indexer.index_path(project.path)
            total = _merged(total, stats)
            _print_index_summary(project.name, stats, verbose=verbose)
        if verbose:
            console.print_json(json.dumps(asdict(total), default=str))
        else:
            console.print(
                f"[bold green]done[/bold green]: +{total.chunks_added} chunks "
                f"across {total.files_indexed} files in {len(targets)} projects"
            )
    finally:
        lex.close()
        vec.close()
    return 0


# ---- indexing + search commands -------------------------------------------


def _cmd_index(args: argparse.Namespace, cfg: Config) -> int:
    verbose = bool(getattr(args, "verbose", False))
    lex, vec, _ = _build_stack(cfg)
    try:
        indexer = _build_indexer(cfg, lex, vec)
        total: IndexStats | None = None
        for path in args.paths:
            indexer.progress_label = Path(path).name or str(path)
            stats = indexer.index_path(path)
            total = stats if total is None else _merged(total, stats)
            _print_index_summary(str(path), stats, verbose=verbose)
        if total is not None:
            if verbose:
                console.print_json(json.dumps({"path": "TOTAL", **asdict(total)}, default=str))
            else:
                console.print(
                    f"[bold green]done[/bold green]: +{total.chunks_added} chunks "
                    f"across {total.files_indexed} files"
                )
    finally:
        lex.close()
        vec.close()
    return 0


def _cmd_query(args: argparse.Namespace, cfg: Config) -> int:
    lex, vec, _ = _build_stack(cfg)
    try:
        search = _build_search(cfg, lex, vec)
        prefix: str | None = None
        project = getattr(args, "project", None)
        if project:
            store = _project_store(cfg)
            hit = store.find(project)
            prefix = hit.path if hit is not None else project
        results = asyncio.run(
            search.aquery(
                args.text,
                args.k,
                rerank=args.rerank,
                source_prefix=prefix,
            )
        )
        for r in results:
            payload = {
                "chunk_id": r.chunk_id,
                "source_path": r.source_path,
                "score": r.score,
                "scores_breakdown": dict(r.scores_breakdown),
                "preview": r.text if args.full else _preview(r.text),
            }
            console.print_json(json.dumps(payload))
    finally:
        lex.close()
        vec.close()
    return 0


def _cmd_serve_api(args: argparse.Namespace, cfg: Config) -> int:
    import uvicorn

    from .api import create_app

    lex, vec, _ = _build_stack(cfg)
    indexer = _build_indexer(cfg, lex, vec)
    search = _build_search(cfg, lex, vec)
    store = _project_store(cfg)
    data_dir = _data_dir(cfg)
    app = create_app(
        search=search,
        indexer=indexer,
        auth_token=cfg.api.auth_token,
        project_store=store,
        watcher_pid_provider=lambda: watcher_running(data_dir),
    )
    try:
        uvicorn.run(
            app,
            host=args.host or cfg.api.host,
            port=args.port or cfg.api.port,
            log_level="info",
        )
    finally:
        lex.close()
        vec.close()
    return 0


def _cmd_serve_mcp(args: argparse.Namespace, cfg: Config) -> int:
    # MCP stdio uses stdout for JSON-RPC. Any extra bytes corrupt the protocol
    # and Claude Desktop / Claude Code fail the handshake with "Failed to
    # connect". FastMCP defaults to a banner + INFO log on stdout, so mute
    # them here and send everything else to stderr.
    import os

    os.environ.setdefault("FASTMCP_SHOW_BANNER", "false")
    os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")
    # Force our own logging to stderr regardless of what configure_logging did.
    for handler in list(logging.getLogger().handlers):
        stream = getattr(handler, "stream", None)
        if stream is not None and stream is sys.stdout:
            handler.stream = sys.stderr

    from .mcp_server import create_server

    lex, vec, _ = _build_stack(cfg)
    indexer = _build_indexer(cfg, lex, vec)
    search = _build_search(cfg, lex, vec)
    server = create_server(
        search=search,
        indexer=indexer,
        project_store=_project_store(cfg),
    )
    try:
        server.run()
    finally:
        lex.close()
        vec.close()
    return 0


def _cmd_watch(args: argparse.Namespace, cfg: Config) -> int:
    from .watcher import IndexWatcher

    store = _project_store(cfg)
    paths = args.paths or store.watched_paths() or [str(p) for p in cfg.watch_paths]
    if not paths:
        console.print("[red]no paths provided and no watched projects[/red]")
        return 2
    lex, vec, _ = _build_stack(cfg)
    indexer = _build_indexer(cfg, lex, vec)
    with IndexWatcher(indexer, paths) as watcher:
        console.print(f"[green]watching[/green] {paths} (Ctrl-C to stop)")
        try:
            import time

            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            watcher.flush()
            lex.close()
            vec.close()
    return 0


# ---- interactive TUI ------------------------------------------------------


def _cmd_bench(args: argparse.Namespace, cfg: Config) -> int:
    """Shell out to `python -m eval.token_bench.run_bench` with passthrough args."""
    import subprocess

    forward = [a for a in args.forward if a != "--"]
    cmd = [sys.executable, "-m", "eval.token_bench.run_bench", *forward]
    return subprocess.run(cmd, check=False).returncode


def _cmd_tui(args: argparse.Namespace, cfg: Config) -> int:
    while True:
        console.rule("[bold cyan]hybrid-search[/bold cyan]")
        _cmd_status(args, cfg)
        console.print()
        console.print("[bold]actions[/bold]")
        console.print(" 1. projects list")
        console.print(" 2. projects add")
        console.print(" 3. projects remove")
        console.print(" 4. projects reindex")
        console.print(" 5. query")
        console.print(" 6. services up / watch")
        console.print(" 7. services down")
        console.print(" 8. run eval")
        console.print(" 0. quit")
        choice = Prompt.ask("choose", default="0")
        if choice in ("0", "q", "quit", "exit"):
            return 0
        try:
            _dispatch_tui_choice(choice, args, cfg)
        except KeyboardInterrupt:
            console.print("[yellow]interrupted[/yellow]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]error:[/red] {exc}")


def _dispatch_tui_choice(choice: str, args: argparse.Namespace, cfg: Config) -> None:
    if choice == "1":
        _cmd_projects_list(argparse.Namespace(), cfg)
    elif choice == "2":
        path = Prompt.ask("path")
        name = Prompt.ask("name", default="")
        watch = Confirm.ask("watch?", default=True)
        do_index = Confirm.ask("index now?", default=True)
        _cmd_projects_add(
            argparse.Namespace(
                path=path,
                name=name or None,
                no_watch=not watch,
                skip_index=not do_index,
            ),
            cfg,
        )
    elif choice == "3":
        target = Prompt.ask("name or path")
        keep = Confirm.ask("keep index data?", default=False)
        _cmd_projects_remove(argparse.Namespace(target=target, keep_index=keep), cfg)
    elif choice == "4":
        target = Prompt.ask("project (blank = all)", default="")
        _cmd_projects_reindex(argparse.Namespace(target=target or None), cfg)
    elif choice == "5":
        text = Prompt.ask("query")
        k = IntPrompt.ask("k", default=cfg.default_result_k)
        full = Confirm.ask("full text?", default=False)
        _cmd_query(
            argparse.Namespace(text=text, k=k, rerank=None, full=full),
            cfg,
        )
    elif choice == "6":
        watch = Confirm.ask("also start watcher?", default=True)
        _cmd_up(argparse.Namespace(watch=watch, config=str(args.config)), cfg)
    elif choice == "7":
        _cmd_down(argparse.Namespace(), cfg)
    elif choice == "8":
        import subprocess

        subprocess.run([sys.executable, "-m", "eval.run_eval", "--fake-embedder"], check=False)
    else:
        console.print(f"[red]unknown choice: {choice}[/red]")


# ---- argument parsing -----------------------------------------------------


_CONFIG_KW: dict = {
    "type": Path,
    "default": None,
    "help": (
        "Path to config.yaml. If omitted, looks at $HYBRID_SEARCH_CONFIG, "
        "then ./config.yaml, then parents of cwd, then the install dir."
    ),
}


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", **_CONFIG_KW)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Chatty output: full stats JSON and INFO-level third-party logs.",
    )


def _build_parser() -> argparse.ArgumentParser:
    # Parent parser so --config is accepted at every subcommand position too.
    common = argparse.ArgumentParser(add_help=False)
    _add_config_arg(common)

    # ``lore`` is the canonical binary name; ``hybrid-search`` stays as a
    # backward-compatible alias. argparse auto-detects which invocation was
    # used based on ``sys.argv[0]`` so help text matches what you typed.
    invoked_as = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "lore"
    prog = "lore" if invoked_as not in {"hybrid-search", "hybrid-index", "hybrid-query"} else invoked_as
    root = argparse.ArgumentParser(prog=prog, parents=[common])
    sub = root.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", parents=[common], help="Bootstrap ~/.lore + print PATH hints")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config.")
    p_init.set_defaults(func=_cmd_init)

    p_up = sub.add_parser("up", parents=[common], help="Start Qdrant; optionally spawn watcher")
    p_up.add_argument("--watch", action="store_true", help="Also start watcher daemon")
    p_up.set_defaults(func=_cmd_up)

    p_down = sub.add_parser("down", parents=[common], help="Stop Qdrant and watcher")
    p_down.set_defaults(func=_cmd_down)

    p_restart = sub.add_parser("restart", parents=[common], help="Stop + start watcher (and Qdrant with --full)")
    p_restart.add_argument("--full", action="store_true", help="Also cycle Qdrant.")
    p_restart.add_argument("--no-watch", action="store_true", help="Do not relaunch the watcher.")
    p_restart.set_defaults(func=_cmd_restart)

    p_status = sub.add_parser("status", parents=[common], help="Show service + index status")
    p_status.set_defaults(func=_cmd_status)

    p_index = sub.add_parser("index", parents=[common], help="Index files/directories")
    p_index.add_argument("paths", nargs="+")
    p_index.set_defaults(func=_cmd_index)

    p_query = sub.add_parser("query", parents=[common], help="Run a one-shot query")
    p_query.add_argument("text")
    p_query.add_argument("-k", type=int, default=None)
    p_query.add_argument("--rerank", action="store_true", default=None)
    p_query.add_argument("--no-rerank", action="store_false", dest="rerank")
    p_query.add_argument("--full", action="store_true", help="Print full chunk text.")
    p_query.add_argument(
        "--project",
        default=None,
        help="Restrict search to a registered project (name) or raw path prefix.",
    )
    p_query.set_defaults(func=_cmd_query, rerank=None)

    p_api = sub.add_parser("serve-api", parents=[common], help="Run the FastAPI HTTP server")
    p_api.add_argument("--host", default=None)
    p_api.add_argument("--port", type=int, default=None)
    p_api.set_defaults(func=_cmd_serve_api)

    p_mcp = sub.add_parser("serve-mcp", parents=[common], help="Run the FastMCP stdio server")
    p_mcp.set_defaults(func=_cmd_serve_mcp)

    p_watch = sub.add_parser("watch", parents=[common], help="Run the incremental file watcher")
    p_watch.add_argument("paths", nargs="*")
    p_watch.set_defaults(func=_cmd_watch)

    p_tui = sub.add_parser("tui", parents=[common], help="Interactive menu")
    p_tui.set_defaults(func=_cmd_tui)

    p_bench = sub.add_parser(
        "bench",
        parents=[common],
        help="Token-usage benchmark (plain / mcp / mcp+caveman)",
    )
    p_bench.add_argument(
        "forward",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to `python -m eval.token_bench.run_bench`",
    )
    p_bench.set_defaults(func=_cmd_bench)

    p_proj = sub.add_parser("projects", parents=[common], help="Manage indexed projects")
    proj_sub = p_proj.add_subparsers(dest="projects_command", required=True)

    p_pl = proj_sub.add_parser("list", parents=[common], help="List projects")
    p_pl.set_defaults(func=_cmd_projects_list)

    p_pa = proj_sub.add_parser("add", parents=[common], help="Add a project")
    p_pa.add_argument("path")
    p_pa.add_argument("--name", default=None)
    p_pa.add_argument("--no-watch", action="store_true")
    p_pa.add_argument("--skip-index", action="store_true")
    p_pa.set_defaults(func=_cmd_projects_add)

    p_paa = proj_sub.add_parser(
        "add-all",
        parents=[common],
        help="Add one project per immediate subfolder of <path>",
    )
    p_paa.add_argument("path")
    p_paa.add_argument("--no-watch", action="store_true")
    p_paa.add_argument("--skip-index", action="store_true")
    p_paa.set_defaults(func=_cmd_projects_add_all)

    p_pr = proj_sub.add_parser("remove", parents=[common], help="Remove a project")
    p_pr.add_argument("target")
    p_pr.add_argument("--keep-index", action="store_true")
    p_pr.set_defaults(func=_cmd_projects_remove)

    p_pri = proj_sub.add_parser("reindex", parents=[common], help="Reindex one or all projects")
    p_pri.add_argument("target", nargs="?", default=None)
    p_pri.set_defaults(func=_cmd_projects_reindex)

    return root


NO_CONFIG_COMMANDS = {"init"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    verbose = bool(getattr(args, "verbose", False))
    try:
        if args.command in NO_CONFIG_COMMANDS:
            configure_logging(verbose=verbose)
            return args.func(args)
        cfg = load_config(args.config)
        configure_logging(verbose=verbose, log_path=cfg.log_path)
        return args.func(args, cfg)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130


def index_cmd() -> int:
    return main(["index"] + sys.argv[1:])


def query_cmd() -> int:
    return main(["query"] + sys.argv[1:])


# ---- helpers --------------------------------------------------------------


def _preview(text: str, *, width: int = 200) -> str:
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + "\u2026"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
