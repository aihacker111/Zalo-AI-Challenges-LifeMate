"""Access to a competition split: samples, tool catalog and the organisers' ``zaic_tools`` runtime.

A split folder looks like ``<split>/{<split>.jsonl, <split>_tools_catalog.json, <split>_environment/, audio/, zaic_tools/}``.
Every split ships its own ``zaic_tools`` version, and all of them are called ``zaic_tools``; :func:`load_zaic_tools`
imports each one under a private module name so that several splits can be used in the same process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

CATALOG_SUFFIX = "_tools_catalog.json"


@dataclass(frozen=True)
class Sample:
    """One line of ``<split>.jsonl``."""

    id: str
    audio: str
    context: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    audio_text: str | None = None
    label: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sample:
        return cls(
            id=d["id"],
            audio=d["audio"],
            context=d.get("context") or {},
            history=d.get("history") or [],
            audio_text=d.get("audio_text"),
            label=d.get("label"),
        )

    @property
    def installed_apps(self) -> list[str]:
        return list((self.context.get("installed_apps") or {}).keys())

    @property
    def installed_tools(self) -> list[str]:
        apps = self.context.get("installed_apps") or {}
        return [t for app in apps.values() for t in app.get("tools", [])]

    @property
    def gold_calls(self) -> list[dict[str, Any]] | None:
        return None if self.label is None else list(self.label.get("tool_calls") or [])

    def user_turn_audio(self) -> list[str]:
        """Relative audio paths of every user turn, history first, the request to answer last."""
        paths = [t["audio"] for t in self.history if t.get("role") == "user" and t.get("audio")]
        return [*paths, self.audio]

    def user_turn_texts(self) -> dict[str, str]:
        """Gold transcripts keyed by audio path (train only)."""
        out = {t["audio"]: t["audio_text"] for t in self.history if t.get("role") == "user" and t.get("audio_text")}
        if self.audio_text:
            out[self.audio] = self.audio_text
        return out


def load_samples(jsonl_path: str | Path) -> list[Sample]:
    """Samples of a ``<split>.jsonl`` file (no catalog or environment needed, e.g. for few-shot retrieval)."""
    with open(jsonl_path, encoding="utf-8") as f:
        return [Sample.from_dict(json.loads(line)) for line in f if line.strip()]


def load_zaic_tools(package_dir: str | Path) -> ModuleType:
    """Import a split's ``zaic_tools`` package under a unique module name (idempotent)."""
    package_dir = Path(package_dir).resolve()
    init = package_dir / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"zaic_tools package not found: {package_dir}")
    name = "zaic_tools_" + hashlib.sha1(str(package_dir).encode()).hexdigest()[:10]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot import {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class Split:
    """A split folder (train, public_test, private_test, or a folder mounted at ``/data``)."""

    def __init__(self, split_dir: str | Path):
        self.dir = Path(split_dir).resolve()
        catalogs = sorted(self.dir.glob(f"*{CATALOG_SUFFIX}"))
        if not catalogs:
            raise FileNotFoundError(f"no *{CATALOG_SUFFIX} in {self.dir}")
        self.catalog_path = catalogs[0]
        self.name = self.catalog_path.name[: -len(CATALOG_SUFFIX)]
        self.jsonl_path = self.dir / f"{self.name}.jsonl"
        self.env_dir = self.dir / f"{self.name}_environment"
        self.zaic_dir = self.dir / "zaic_tools"

    @classmethod
    def resolve(cls, data: str | Path, split: str | None = None) -> Split:
        """``data`` is either a split folder itself or a folder that contains ``<split>/``."""
        root = Path(data)
        if any(root.glob(f"*{CATALOG_SUFFIX}")):
            return cls(root)
        if split is None:
            raise FileNotFoundError(f"{root} is not a split folder; pass split=")
        return cls(root / split)

    def __repr__(self) -> str:
        return f"Split(name={self.name!r}, dir={str(self.dir)!r})"

    # -- samples -------------------------------------------------------------------------------
    @cached_property
    def samples(self) -> list[Sample]:
        return load_samples(self.jsonl_path)

    @cached_property
    def _index(self) -> dict[str, Sample]:
        return {s.id: s for s in self.samples}

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, sample_id: str) -> Sample:
        return self._index[sample_id]

    @property
    def has_labels(self) -> bool:
        return bool(self.samples) and self.samples[0].label is not None

    def audio_path(self, rel: str) -> Path:
        return self.dir / rel

    # -- catalog -------------------------------------------------------------------------------
    @cached_property
    def catalog(self) -> dict[str, Any]:
        with open(self.catalog_path, encoding="utf-8") as f:
            return json.load(f)

    @cached_property
    def apps(self) -> dict[str, dict[str, Any]]:
        return {a["id"]: a for a in self.catalog.get("apps", [])}

    @cached_property
    def tools(self) -> dict[str, dict[str, Any]]:
        return {t["name"]: t for a in self.catalog.get("apps", []) for t in a.get("tools", [])}

    # -- runtime -------------------------------------------------------------------------------
    @cached_property
    def zaic(self) -> ModuleType:
        return load_zaic_tools(self.zaic_dir)

    @cached_property
    def executor(self) -> Any:
        """The organisers' ``Executor`` over this split (loads every initial state once)."""
        return self.zaic.Executor(str(self.env_dir), str(self.catalog_path))

    def open_session(self, sample_id: str) -> Any:
        """A fresh session on the initial state of ``sample_id`` — the object ``ToolEnv`` returns.

        Inference must open exactly ONE session per sample: every call made on it is part of the submission.
        """
        return self.executor.open(sample_id)
