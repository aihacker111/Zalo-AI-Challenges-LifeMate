"""Deterministic checks between the model and ``env.call``.

Every call made on the session is part of the submission, and a rejected or redundant call lowers precision. The guard
fixes what code can fix (types, date-time format, unknown or empty arguments) and returns to the model, *without*
calling the environment, anything that would be a schema error, a repeated call or a write on an ID never read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lifemate.agent.parser import Action
from lifemate.agent.render import compact_result
from lifemate.catalog import ToolSpec
from lifemate.config import GuardConfig
from lifemate.timeutil import normalize_datetime

_EMPTY = (None, "", [], {})
_INT_TEXT = re.compile(r"^[+-]?\d{1,3}([.,\s]\d{3})+$|^[+-]?\d+$")


def collect_strings(obj: Any, out: set[str]) -> set[str]:
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_strings(v, out)
    return out


def _key(action: Action) -> str:
    return action.name + json.dumps(action.arguments, sort_keys=True, ensure_ascii=False)


def coerce_value(value: Any, p: dict[str, Any], offset: str) -> Any:
    t = p.get("type")
    if t == "integer":
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and _INT_TEXT.match(value.strip()):
            return int(re.sub(r"[.,\s]", "", value.strip()))
    elif t == "number":
        if isinstance(value, str):
            try:
                return float(value.replace(",", "."))
            except ValueError:
                return value
    elif t == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and p.get("format") == "date-time":
            return normalize_datetime(value, offset)
    elif t == "boolean" and isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    elif t == "array":
        if not isinstance(value, list) and value not in _EMPTY:
            value = [value]
        if isinstance(value, list) and isinstance(p.get("items"), dict):
            return [coerce_value(v, p["items"], offset) for v in value]
    return value


@dataclass
class Decision:
    execute: bool
    action: Action
    message: str = ""
    cached: Any = None


@dataclass
class Guard:
    session: Any
    specs: dict[str, ToolSpec]
    allowed: set[str]
    offset: str
    cfg: GuardConfig = field(default_factory=GuardConfig)
    seen: set[str] = field(default_factory=set)
    _reads: dict[str, Any] = field(default_factory=dict)
    _writes: set[str] = field(default_factory=set)
    _fails: int = 0

    @classmethod
    def for_sample(cls, session: Any, specs: list[ToolSpec], history: list[dict[str, Any]], offset: str, cfg: GuardConfig) -> Guard:
        seen: set[str] = set()
        for turn in history:
            for c in turn.get("tool_calls") or []:
                collect_strings(c.get("arguments"), seen)
            collect_strings(turn.get("tool_results"), seen)
        return cls(session=session, specs={s.name: s for s in specs}, allowed={s.name for s in specs}, offset=offset, cfg=cfg, seen=seen)

    # -- argument repair ----------------------------------------------------------------------
    def repair(self, action: Action) -> Action:
        spec = self.specs.get(action.name)
        if spec is None:
            return action
        args = dict(action.arguments)
        props = spec.properties
        if self.cfg.drop_unknown_args:
            args = {k: v for k, v in args.items() if k in props}
        if self.cfg.drop_empty_optional_args:
            args = {k: v for k, v in args.items() if v not in _EMPTY or k in spec.required}
        if self.cfg.coerce_types:
            args = {k: coerce_value(v, props.get(k, {}), self.offset) for k, v in args.items()}
        return Action(action.name, args)

    # -- review -------------------------------------------------------------------------------
    def review(self, action: Action) -> Decision:
        if action.name not in self.allowed:
            return Decision(False, action, f"tool '{action.name}' không có trong danh sách app đã cài")
        action = self.repair(action)
        spec = self.specs[action.name]
        key = _key(action)
        if spec.is_read and self.cfg.block_duplicate_reads and key in self._reads:
            return Decision(False, action, f"đã gọi {action.name} với cùng tham số, dùng lại kết quả trước", self._reads[key])
        if spec.is_write and self.cfg.block_duplicate_writes and key in self._writes:
            return Decision(False, action, f"{action.name} với cùng tham số đã được thực hiện")
        if self.cfg.require_seen_refs and spec.kind != "meta":
            for param in spec.ref_params:
                values = action.arguments.get(param)
                values = values if isinstance(values, list) else [values]
                missing = [v for v in values if isinstance(v, str) and v and v not in self.seen]
                if missing:
                    hint = spec.properties[param].get("description", "")
                    return Decision(False, action, f"ID {missing[0]!r} của {param} chưa xuất hiện trong kết quả tool nào. Hãy gọi tool đọc để lấy ID trước ({hint}).")
        if self.cfg.validate_before_call:
            check = self.session.validate(action.name, action.arguments)
            if not check.get("ok", False):
                self._fails += 1
                return Decision(False, action, f"sai schema: {check.get('error')}. Sửa lại tham số.")
        return Decision(True, action)

    def record(self, action: Action, result: Any) -> None:
        spec = self.specs.get(action.name)
        key = _key(action)
        if spec is not None and spec.is_read:
            self._reads[key] = result
        elif spec is not None and spec.is_write:
            self._writes.add(key)
        collect_strings(result, self.seen)

    def cached_text(self, decision: Decision, max_chars: int) -> str:
        return compact_result(decision.cached, max_chars=max_chars) if decision.cached is not None else ""
