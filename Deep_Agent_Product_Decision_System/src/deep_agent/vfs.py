"""Shared Virtual File System (VFS).

A path-addressed, in-memory artifact store through which agents exchange
Pydantic-validated JSON. Keeping large artifacts out of the graph state
keeps LangGraph checkpoints small and makes every intermediate step
auditable (`vfs.tree()` shows the full run trace).

Optionally mirrors writes to a real directory for debugging.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class VirtualFileSystem:
    def __init__(self, mirror_dir: Optional[str] = None) -> None:
        self._files: dict[str, str] = {}
        self._lock = threading.Lock()
        self._mirror = Path(mirror_dir) if mirror_dir else None
        if self._mirror:
            self._mirror.mkdir(parents=True, exist_ok=True)

    # -- raw ----------------------------------------------------------------
    def write(self, path: str, content: str) -> str:
        path = self._norm(path)
        with self._lock:
            self._files[path] = content
        if self._mirror:
            dest = self._mirror / path.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        return path

    def read(self, path: str) -> str:
        path = self._norm(path)
        with self._lock:
            if path not in self._files:
                raise FileNotFoundError(f"VFS: no such file {path}")
            return self._files[path]

    def exists(self, path: str) -> bool:
        with self._lock:
            return self._norm(path) in self._files

    def ls(self, prefix: str = "/") -> list[str]:
        prefix = self._norm(prefix)
        with self._lock:
            return sorted(p for p in self._files if p.startswith(prefix))

    def tree(self) -> str:
        return "\n".join(self.ls())

    # -- pydantic-validated -------------------------------------------------
    def write_model(self, path: str, model: BaseModel) -> str:
        return self.write(path, model.model_dump_json(indent=2))

    def read_model(self, path: str, model_cls: Type[T]) -> T:
        return model_cls.model_validate_json(self.read(path))

    def write_json(self, path: str, obj) -> str:
        return self.write(path, json.dumps(obj, indent=2))

    def read_json(self, path: str):
        return json.loads(self.read(path))

    @staticmethod
    def _norm(path: str) -> str:
        return "/" + path.strip("/")
