"""Tests for ProjectStore."""

from __future__ import annotations

from pathlib import Path

from hybrid_search.projects import Project, ProjectStore


def test_empty_store_returns_empty(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    assert store.load() == []


def test_add_and_load_roundtrip(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    target = tmp_path / "proj-a"
    target.mkdir()
    project = store.add(target, name="alpha", watch=True)
    assert project.name == "alpha"
    assert project.path == str(target.resolve())
    assert project.watch
    assert len(store.load()) == 1


def test_add_duplicate_updates_in_place(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    target = tmp_path / "proj"
    target.mkdir()
    store.add(target, name="p", watch=True)
    store.add(target, name="p", watch=False)
    projects = store.load()
    assert len(projects) == 1
    assert projects[0].watch is False


def test_add_dedup_by_path_with_unique_name(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    a = tmp_path / "a"
    b = tmp_path / "proj"
    a.mkdir()
    b.mkdir()
    store.add(a, name="proj")
    store.add(b)  # would otherwise collide on name
    projects = store.load()
    names = {p.name for p in projects}
    assert len(names) == 2  # unique names preserved


def test_remove_by_name_and_path(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    store.add(a, name="alpha")
    store.add(b, name="beta")
    assert store.remove("alpha") is not None
    assert len(store.load()) == 1
    assert store.remove(str(b)) is not None
    assert store.load() == []


def test_remove_missing_returns_none(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    assert store.remove("nope") is None


def test_set_watch_toggles(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    target = tmp_path / "p"
    target.mkdir()
    store.add(target, name="p", watch=True)
    store.set_watch("p", False)
    assert store.load()[0].watch is False


def test_watched_paths_filters(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    store.add(a, name="a", watch=True)
    store.add(b, name="b", watch=False)
    assert store.watched_paths() == [str(a.resolve())]


def test_corrupt_json_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    path.write_text("not json", encoding="utf-8")
    assert ProjectStore(path).load() == []


def test_project_dataclass_roundtrip() -> None:
    p = Project(name="x", path="/tmp/x", watch=False, added_at=1.0)
    assert p.name == "x"
    assert not p.watch
