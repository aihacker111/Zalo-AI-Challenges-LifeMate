"""Tool specifications: compact rendering for prompts and JSON schemas for constrained decoding."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

FINISH = "finish"
ASK_TOOL = "sys.ask_missing_info"
KIND_VI = {"read": "đọc", "write": "ghi", "meta": "hỏi lại"}
_DECODING_KEYS = {"type", "enum", "items", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    app: str
    kind: str
    description: str
    parameters: dict[str, Any]
    returns: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolSpec:
        return cls(
            name=d["name"],
            app=d.get("app") or d["name"].split(".", 1)[0],
            kind=d.get("kind", "write"),
            description=d.get("description", ""),
            parameters=d.get("parameters") or {"type": "object", "properties": {}},
            returns=d.get("returns") or "",
        )

    @property
    def properties(self) -> dict[str, dict[str, Any]]:
        return self.parameters.get("properties") or {}

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required") or [])

    @property
    def one_of(self) -> list[list[str]]:
        return [list(g) for g in self.parameters.get("x-one-of") or []]

    @property
    def fuzzy_params(self) -> set[str]:
        return {k for k, p in self.properties.items() if p.get("x-match") == "fuzzy"}

    @property
    def ref_params(self) -> dict[str, str]:
        return {k: p["x-ref"] for k, p in self.properties.items() if p.get("x-ref")}

    @property
    def is_read(self) -> bool:
        return self.kind == "read"

    @property
    def is_write(self) -> bool:
        return self.kind == "write"


def load_specs(tools: dict[str, dict[str, Any]]) -> dict[str, ToolSpec]:
    return {name: ToolSpec.from_dict(t) for name, t in tools.items()}


# -- prompt rendering ---------------------------------------------------------------------------
def _type_str(p: dict[str, Any]) -> str:
    if p.get("enum"):
        return "|".join(f'"{v}"' if isinstance(v, str) else str(v) for v in p["enum"])
    t = p.get("type", "any")
    if t == "array":
        item = p.get("items") or {}
        return f"list[{_type_str(item)}]" if item else "list"
    if p.get("format"):
        return f"{t}<{p['format']}>"
    return str(t)


def _bounds(p: dict[str, Any]) -> str:
    parts = []
    for key, sym in (("minimum", ">="), ("exclusiveMinimum", ">"), ("maximum", "<="), ("exclusiveMaximum", "<")):
        if key in p:
            parts.append(f"{sym}{p[key]}")
    return " ".join(parts)


def render_param(name: str, p: dict[str, Any], required: bool, max_desc: int = 110) -> str:
    flags = [_type_str(p)]
    if b := _bounds(p):
        flags.append(b)
    if p.get("x-match") == "fuzzy":
        flags.append("văn bản tự do")
    line = f"{name}{'*' if required else ''}: {', '.join(flags)}"
    desc = (p.get("description") or "").strip()
    if len(desc) > max_desc:
        desc = desc[: max_desc - 1].rstrip() + "…"
    return f"{line} — {desc}" if desc else line


def render_tool(spec: ToolSpec) -> str:
    """Compact, model-facing description of one tool (about a third of the raw JSON schema)."""
    params = ", ".join(f"{k}{'*' if k in spec.required else ''}" for k in spec.properties)
    head = f"- {spec.name}({params}) [{KIND_VI.get(spec.kind, spec.kind)}] {spec.description}".rstrip()
    if spec.returns:
        head += f" → {spec.returns}"
    lines = [head]
    for k, p in spec.properties.items():
        lines.append("    " + render_param(k, p, k in spec.required))
    if spec.one_of:
        lines.append("    bắt buộc có một trong: " + " | ".join(" + ".join(g) for g in spec.one_of))
    return "\n".join(lines)


def render_toolbox(specs: list[ToolSpec], apps: dict[str, dict[str, Any]] | None = None) -> str:
    """Tools grouped by app, in the order given."""
    apps = apps or {}
    groups: dict[str, list[ToolSpec]] = {}
    for s in specs:
        groups.setdefault(s.app, []).append(s)
    blocks = []
    for app, tools in groups.items():
        meta = apps.get(app, {})
        title = f"## {app}"
        if meta.get("name_vi") or meta.get("description"):
            title += f" — {meta.get('name_vi', '')}: {meta.get('description', '')}".rstrip(": ")
        blocks.append("\n".join([title, *(render_tool(t) for t in tools)]))
    return "\n\n".join(blocks)


# -- constrained decoding -----------------------------------------------------------------------
def _clean(p: dict[str, Any]) -> dict[str, Any]:
    out = {k: copy.deepcopy(v) for k, v in p.items() if k in _DECODING_KEYS}
    if isinstance(out.get("items"), dict):
        out["items"] = _clean(out["items"])
    return out


def arguments_schema(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: _clean(p) for k, p in spec.properties.items()},
        "required": spec.required,
        "additionalProperties": False,
    }


def action_schema(specs: list[ToolSpec]) -> dict[str, Any]:
    """JSON schema of one agent step: a call to one of ``specs`` or ``{"name": "finish"}``."""
    variants = [
        {
            "type": "object",
            "properties": {"name": {"const": s.name}, "arguments": arguments_schema(s)},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for s in specs
    ]
    variants.append(
        {
            "type": "object",
            "properties": {"name": {"const": FINISH}},
            "required": ["name"],
            "additionalProperties": False,
        }
    )
    return {"anyOf": variants}
