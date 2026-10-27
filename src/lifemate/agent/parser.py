"""Turns raw model text into an action."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lifemate.catalog import FINISH

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FINISH_WORDS = {"finish", "done", "stop", "end"}


@dataclass
class Action:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def is_finish(self) -> bool:
        return self.name == FINISH


class ParseError(ValueError):
    pass


def _first_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> Action:
    raw = _THINK.sub("", text or "").strip()
    if m := _TOOL_CALL.search(raw):
        raw = m.group(1)
    elif m := _FENCE.search(raw):
        raw = m.group(1)
    if raw.strip().strip('"').lower() in _FINISH_WORDS:
        return Action(FINISH)
    blob = _first_object(raw)
    if blob is None:
        raise ParseError(f"không tìm thấy JSON trong: {text[:200]!r}")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON lỗi: {e}") from e
    name = obj.get("name") or obj.get("tool") or obj.get("action")
    if not isinstance(name, str):
        raise ParseError("thiếu trường name")
    if name.lower() in _FINISH_WORDS:
        return Action(FINISH)
    args = obj.get("arguments", obj.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            raise ParseError(f"arguments không phải JSON: {e}") from e
    if not isinstance(args, dict):
        raise ParseError("arguments phải là object")
    return Action(name, args)
