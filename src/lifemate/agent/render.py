"""Canonical text forms shared by inference and training data (they must never diverge)."""

from __future__ import annotations

import copy
import json
from typing import Any

from lifemate.catalog import FINISH


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def format_action(name: str, arguments: dict[str, Any] | None = None) -> str:
    if name == FINISH:
        return dumps({"name": FINISH})
    return dumps({"name": name, "arguments": arguments or {}})


def _shrink(value: Any, max_items: int) -> Any:
    if isinstance(value, list):
        items = [_shrink(v, max_items) for v in value[:max_items]]
        if len(value) > max_items:
            items.append(f"…(+{len(value) - max_items} mục)")
        return items
    if isinstance(value, dict):
        return {k: _shrink(v, max_items) for k, v in value.items()}
    return value


def compact_result(result: Any, max_items: int = 15, max_chars: int = 1500) -> str:
    text = dumps(_shrink(copy.deepcopy(result), max_items))
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def format_observation(name: str, result: Any, max_items: int = 15, max_chars: int = 1500) -> str:
    return f"Kết quả {name}: {compact_result(result, max_items, max_chars)}"


def format_rejection(message: str) -> str:
    return f"Không thực hiện: {message}"
