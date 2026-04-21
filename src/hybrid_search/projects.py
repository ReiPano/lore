"""Persistent list of indexed projects.

Each project is a named absolute path the user has asked the tool to index
(and optionally watch). The store lives in a JSON file next to the lexical
index by default so it travels with the data dir.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Project:
    name: str
    path: str
    watch: bool = True
    added_at: float = field(default_factory=time.time)


class ProjectStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ---- persistence -------------------------------------------------

    def load(self) -> list[Project]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        return [
            Project(
                name=str(item["name"]),
                path=str(item["path"]),
                watch=bool(item.get("watch", True)),
                added_at=float(item.get("added_at", time.time())),
            )
            for item in raw
            if isinstance(item, dict) and "name" in item and "path" in item
        ]

    def save(self, projects: list[Project]) -> None:
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(p) for p in projects]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---- mutations ---------------------------------------------------

    def add(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        watch: bool = True,
    ) -> Project:
        resolved = str(Path(path).expanduser().resolve())
        effective_name = name or Path(resolved).name or "project"
        projects = self.load()
        for existing in projects:
            if existing.path == resolved:
                existing.watch = watch
                if name:
                    existing.name = name
                self.save(projects)
                return existing
        # Ensure unique name
        taken = {p.name for p in projects}
        final_name = effective_name
        n = 1
        while final_name in taken:
            n += 1
            final_name = f"{effective_name}-{n}"
        project = Project(name=final_name, path=resolved, watch=watch)
        projects.append(project)
        self.save(projects)
        return project

    def remove(self, name_or_path: str) -> Project | None:
        projects = self.load()
        target = self._match(projects, name_or_path)
        if target is None:
            return None
        remaining = [p for p in projects if p is not target]
        self.save(remaining)
        return target

    def set_watch(self, name_or_path: str, watch: bool) -> Project | None:
        projects = self.load()
        target = self._match(projects, name_or_path)
        if target is None:
            return None
        target.watch = watch
        self.save(projects)
        return target

    def find(self, name_or_path: str) -> Project | None:
        return self._match(self.load(), name_or_path)

    def watched_paths(self) -> list[str]:
        return [p.path for p in self.load() if p.watch]

    # ---- internals ---------------------------------------------------

    @staticmethod
    def _match(projects: list[Project], key: str) -> Project | None:
        for p in projects:
            if p.name == key:
                return p
        resolved = str(Path(key).expanduser().resolve())
        for p in projects:
            if p.path == resolved:
                return p
        return None
