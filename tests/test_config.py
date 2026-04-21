"""Tests for config loading + resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_search.config import (
    CONFIG_ENV_VAR,
    LEGACY_CONFIG_ENV_VAR,
    USER_CONFIG_PATH,
    load_config,
    resolve_config_path,
)


def _write_config(path: Path, **overrides: str) -> Path:
    defaults = {
        "index_path": "/tmp/lore-test/lex.sqlite3",
        "qdrant_url": "http://localhost:6333",
        "collection_name": "lore_test",
        "embedding_model": "fake-hash-v1",
        "embedding_dim": "32",
        "chunk_size": "500",
        "chunk_overlap": "50",
    }
    defaults.update(overrides)
    lines = [f"{k}: {v}" for k, v in defaults.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_config_exposes_exclude_content_patterns(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "index_path: /tmp/lore-test/lex.sqlite3\n"
        "qdrant_url: http://localhost:6333\n"
        "collection_name: lore_test\n"
        "embedding_model: fake-hash-v1\n"
        "embedding_dim: 32\n"
        "chunk_size: 500\n"
        "chunk_overlap: 50\n"
        "exclude_content_patterns:\n"
        "  - \"sk-ant-[A-Za-z0-9]+\"\n"
        "  - \"AKIA[0-9A-Z]{16}\"\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.exclude_content_patterns == [
        "sk-ant-[A-Za-z0-9]+",
        "AKIA[0-9A-Z]{16}",
    ]


def test_load_config_defaults_exclude_content_patterns_to_empty(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load_config(cfg_path)
    assert cfg.exclude_content_patterns == []


def test_resolve_config_path_prefers_explicit(tmp_path: Path) -> None:
    explicit = _write_config(tmp_path / "explicit.yaml")
    resolved = resolve_config_path(explicit)
    assert resolved == explicit.resolve()


def test_resolve_config_path_env_var_preferred_over_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_cfg = _write_config(tmp_path / "via-env.yaml")
    # Clear both env vars and user dir, then set the canonical one.
    monkeypatch.setenv(CONFIG_ENV_VAR, str(env_cfg))
    monkeypatch.delenv(LEGACY_CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    # Point the user-level fallback at a missing path to avoid shadowing.
    monkeypatch.setattr("hybrid_search.config.USER_CONFIG_PATH", tmp_path / "absent.yaml")
    monkeypatch.setattr(
        "hybrid_search.config.LEGACY_USER_CONFIG_PATH", tmp_path / "absent-legacy.yaml"
    )
    resolved = resolve_config_path(None)
    assert resolved == env_cfg.resolve()


def test_resolve_config_path_legacy_env_var_still_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_cfg = _write_config(tmp_path / "legacy-env.yaml")
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setenv(LEGACY_CONFIG_ENV_VAR, str(legacy_cfg))
    monkeypatch.setattr("hybrid_search.config.USER_CONFIG_PATH", tmp_path / "absent.yaml")
    monkeypatch.setattr(
        "hybrid_search.config.LEGACY_USER_CONFIG_PATH", tmp_path / "absent-legacy.yaml"
    )
    resolved = resolve_config_path(None)
    assert resolved == legacy_cfg.resolve()


def test_resolve_config_path_errors_cleanly_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hybrid_search.config.USER_CONFIG_PATH", tmp_path / "absent.yaml")
    monkeypatch.setattr(
        "hybrid_search.config.LEGACY_USER_CONFIG_PATH", tmp_path / "absent-legacy.yaml"
    )
    # Also block the walk-up fallback into the repo root by isolating tmp_path.
    with pytest.raises(FileNotFoundError):
        resolve_config_path(tmp_path / "does-not-exist.yaml")


def test_user_config_path_points_at_lore_dir() -> None:
    assert str(USER_CONFIG_PATH).endswith(".lore/config.yaml")