from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")
CONFIG_ENV_VAR = "LORE_CONFIG"
LEGACY_CONFIG_ENV_VAR = "HYBRID_SEARCH_CONFIG"

# Tunable weight/chunk presets keyed by ``profile`` in config.yaml. User
# fields in the same file still win — the preset is applied first so any
# explicit value overrides it.
PROFILES: dict[str, dict[str, object]] = {
    "mixed": {
        "bm25_weight": 1.0,
        "vector_weight": 1.0,
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "code": {
        "bm25_weight": 2.0,
        "vector_weight": 1.0,
        "chunk_size": 400,
        "chunk_overlap": 40,
    },
    "docs": {
        "bm25_weight": 1.0,
        "vector_weight": 1.5,
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "notes": {
        "bm25_weight": 1.0,
        "vector_weight": 2.0,
        "chunk_size": 300,
        "chunk_overlap": 30,
    },
}

USER_DATA_DIR = Path("~/.lore").expanduser()
USER_CONFIG_PATH = USER_DATA_DIR / "config.yaml"

# Backwards-compat: the old default. Still honored for discovery so
# existing installs keep working until the user migrates data.
LEGACY_USER_DATA_DIR = Path("~/.better-mem").expanduser()
LEGACY_USER_CONFIG_PATH = LEGACY_USER_DATA_DIR / "config.yaml"


def _p(value) -> Path:
    """Expand ``~`` and environment variables in a path string."""
    return Path(os.path.expandvars(str(value))).expanduser()


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Find the config.yaml the user meant.

    Precedence:
      1. Explicit ``path`` argument.
      2. ``HYBRID_SEARCH_CONFIG`` environment variable.
      3. ``~/.lore/config.yaml`` (created by ``lore init``). Falls back to
         ``~/.better-mem/config.yaml`` if the new path does not exist yet.
      4. ``./config.yaml`` in the current working directory; walk up.
      5. ``config.yaml`` next to the installed package (dev checkout).
    """

    if path is not None:
        p = _p(path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"config not found: {p}")

    for env_name in (CONFIG_ENV_VAR, LEGACY_CONFIG_ENV_VAR):
        env_value = os.environ.get(env_name)
        if env_value:
            env_path = _p(env_value)
            if env_path.exists():
                return env_path.resolve()

    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH.resolve()
    if LEGACY_USER_CONFIG_PATH.exists():
        return LEGACY_USER_CONFIG_PATH.resolve()

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate.resolve()

    package_root = Path(__file__).resolve().parents[2]
    pkg_candidate = package_root / "config.yaml"
    if pkg_candidate.exists():
        return pkg_candidate.resolve()

    raise FileNotFoundError(
        "no config.yaml found. Run `lore init`, pass --config, or set $LORE_CONFIG."
    )


@dataclass(slots=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token: str | None = None


@dataclass(slots=True)
class Config:
    index_path: Path
    qdrant_url: str
    collection_name: str
    embedding_model: str
    embedding_dim: int
    rerank_model: str
    rerank_enabled: bool
    chunk_size: int
    chunk_overlap: int
    fusion_k: int
    bm25_weight: float
    vector_weight: float
    retrieval_k_per_index: int
    rerank_top_n: int
    default_result_k: int
    watch_paths: list[Path] = field(default_factory=list)
    max_file_bytes: int = 10 * 1024 * 1024
    supported_extensions: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    exclude_content_patterns: list[str] = field(default_factory=list)
    api: ApiConfig = field(default_factory=ApiConfig)
    log_path: Path | None = None


def load_config(path: str | Path | None = None) -> Config:
    resolved = resolve_config_path(path)
    raw = _read_yaml(resolved)
    profile_name = raw.get("profile")
    if profile_name:
        preset = PROFILES.get(str(profile_name).lower())
        if preset is None:
            raise ValueError(
                f"unknown profile {profile_name!r}; "
                f"valid options: {sorted(PROFILES)}"
            )
        # User-provided keys win; presets only fill in anything the user
        # didn't specify explicitly.
        for key, value in preset.items():
            raw.setdefault(key, value)
    api_raw: dict[str, Any] = raw.get("api") or {}
    return Config(
        index_path=_p(raw["index_path"]),
        qdrant_url=raw["qdrant_url"],
        collection_name=raw["collection_name"],
        embedding_model=raw["embedding_model"],
        embedding_dim=int(raw["embedding_dim"]),
        rerank_model=raw.get("rerank_model", ""),
        rerank_enabled=bool(raw.get("rerank_enabled", False)),
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        fusion_k=int(raw.get("fusion_k", 60)),
        bm25_weight=float(raw.get("bm25_weight", 1.0)),
        vector_weight=float(raw.get("vector_weight", 1.0)),
        retrieval_k_per_index=int(raw.get("retrieval_k_per_index", 50)),
        rerank_top_n=int(raw.get("rerank_top_n", 20)),
        default_result_k=int(raw.get("default_result_k", 10)),
        watch_paths=[_p(p) for p in raw.get("watch_paths") or []],
        max_file_bytes=int(raw.get("max_file_bytes", 10 * 1024 * 1024)),
        supported_extensions=list(raw.get("supported_extensions") or []),
        exclude_dirs=list(raw.get("exclude_dirs") or []),
        exclude_patterns=list(raw.get("exclude_patterns") or []),
        exclude_content_patterns=list(raw.get("exclude_content_patterns") or []),
        api=ApiConfig(
            host=api_raw.get("host", "127.0.0.1"),
            port=int(api_raw.get("port", 8765)),
            auth_token=api_raw.get("auth_token"),
        ),
        log_path=_p(raw["log_path"]) if raw.get("log_path") else None,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data
