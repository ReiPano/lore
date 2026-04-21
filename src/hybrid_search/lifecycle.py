"""Process and service lifecycle helpers for the CLI.

- Qdrant: managed via ``docker compose`` commands run from the project root.
- Watcher: spawned as a detached child process. Its PID is written to a file
  alongside the lexical index so subsequent CLI invocations can check status,
  stop it, or refuse to start a second copy.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ---- Qdrant ----------------------------------------------------------------


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=capture,
        text=True,
        check=False,
        env=env,
    )


def _compose_env(qdrant_data: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if qdrant_data is not None:
        resolved = str(qdrant_data.expanduser().resolve())
        env["LORE_QDRANT_DATA"] = resolved
        env["HYBRID_QDRANT_DATA"] = resolved  # backwards-compat alias
    return env


def _compose_dir(compose_file: Path | None) -> Path:
    if compose_file:
        return compose_file.resolve().parent
    # Walk up from CWD first so projects with their own compose win.
    cur = Path.cwd()
    for candidate in [cur, *cur.parents]:
        if (candidate / "docker-compose.yml").exists():
            return candidate
    # Fall back to the install dir (repo root where lifecycle.py lives).
    # lifecycle.py -> src/hybrid_search/lifecycle.py; repo root is two levels up.
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "docker-compose.yml").exists():
        return package_root
    return cur


def qdrant_up(*, compose_file: Path | None = None, qdrant_data: Path | None = None) -> tuple[bool, str]:
    cwd = _compose_dir(compose_file)
    if not (cwd / "docker-compose.yml").exists():
        return False, f"no docker-compose.yml found (searched from {Path.cwd()})"
    env = _compose_env(qdrant_data)
    proc = _run(["docker", "compose", "up", "-d"], cwd=cwd, capture=True, env=env)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return True, proc.stdout.strip() or "qdrant up"


def qdrant_down(*, compose_file: Path | None = None, qdrant_data: Path | None = None) -> tuple[bool, str]:
    cwd = _compose_dir(compose_file)
    env = _compose_env(qdrant_data)
    proc = _run(["docker", "compose", "down"], cwd=cwd, capture=True, env=env)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return True, "qdrant down"


def qdrant_status(*, compose_file: Path | None = None, qdrant_data: Path | None = None) -> dict[str, str]:
    cwd = _compose_dir(compose_file)
    env = _compose_env(qdrant_data)
    proc = _run(
        ["docker", "compose", "ps", "--format", "json"],
        cwd=cwd,
        capture=True,
        env=env,
    )
    if proc.returncode != 0:
        return {"running": "unknown", "detail": proc.stderr.strip() or "docker compose unavailable"}
    out = proc.stdout.strip()
    if not out:
        return {"running": "no", "detail": "no containers"}
    return {"running": "yes", "detail": out.splitlines()[0][:200]}


# ---- Watcher daemon --------------------------------------------------------


@dataclass(slots=True)
class WatcherHandle:
    pid: int
    pid_file: Path


def watcher_pid_file(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "watcher.pid"


def watcher_running(data_dir: Path) -> int | None:
    pid_file = watcher_pid_file(data_dir)
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if not _pid_alive(pid):
        try:
            pid_file.unlink()
        except OSError:
            pass
        return None
    return pid


def watcher_start(
    *,
    data_dir: Path,
    config_path: Path,
    paths: list[str],
    log_path: Path | None = None,
) -> WatcherHandle:
    existing = watcher_running(data_dir)
    if existing is not None:
        raise RuntimeError(f"watcher already running (pid {existing})")
    log_file = log_path or (data_dir / "watcher.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "ab", buffering=0)
    cmd = [
        sys.executable,
        "-m",
        "hybrid_search",
        "--config",
        str(config_path),
        "watch",
        *paths,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=fh,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file = watcher_pid_file(data_dir)
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    return WatcherHandle(pid=proc.pid, pid_file=pid_file)


def watcher_stop(data_dir: Path, *, timeout: float = 5.0) -> bool:
    pid = watcher_running(data_dir)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        watcher_pid_file(data_dir).unlink(missing_ok=True)
        return False
    # Poll until it dies or timeout.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    watcher_pid_file(data_dir).unlink(missing_ok=True)
    return True


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
