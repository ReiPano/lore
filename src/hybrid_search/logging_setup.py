"""Shared logging configuration.

Two handlers are installed:
- ``stderr`` — at ``WARNING`` by default so the CLI stays quiet; at ``INFO``
  when ``verbose`` is true.
- Rotating log file — always at ``INFO`` so the on-disk log stays useful for
  debugging regardless of verbosity.

Noisy third-party loggers (httpx, urllib3, qdrant_client, fastembed, etc.)
are muted on stderr unless ``verbose`` is set.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "qdrant_client",
    "fastembed",
    "watchdog",
)


def configure_logging(
    *,
    verbose: bool = False,
    log_path: str | Path | None = None,
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers decide visibility

    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.INFO if verbose else logging.WARNING)
    stderr.setFormatter(fmt)
    root.addHandler(stderr)

    if log_path:
        p = Path(log_path).expanduser()
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            p,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Mute third-party chatter on stderr unless explicitly asked.
    noisy_level = logging.INFO if verbose else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)
