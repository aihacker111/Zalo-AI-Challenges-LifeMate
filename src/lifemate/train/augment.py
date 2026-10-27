"""Tool-renaming augmentation: teaches the model to read the schema instead of memorising tool names.

The private test brings apps the model has never seen. Renaming apps, tools and parameters consistently (in the tool
list, the history and the gold calls, including ``sys.ask_missing_info`` arguments) produces new surface forms of the
same trajectories at no labelling cost. IDs and tool results are left untouched.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any

from lifemate.catalog import ASK_TOOL, ToolSpec
from lifemate.data import Sample


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


class Renamer:
    def __init__(self, specs: list[ToolSpec], rng: random.Random):
        self.app_map: dict[str, str] = {}
        self.tool_map: dict[str, str] = {}
        self.param_map: dict[str, dict[str, str]] = {}
        used: set[str] = set()
        for s in specs:
            if s.name == ASK_TOOL:
                continue
            if s.app not in self.app_map:
                parts = s.app.split("_")
                choices = [s.app.replace("_", ""), parts[-1], f"{s.app}_app", "".join(p[:4] for p in parts)]
                new = next((c for c in rng.sample(choices, len(choices)) if c not in used and c != "sys"), f"{s.app}_x")
                used.add(new)
                self.app_map[s.app] = new
            tool = s.name.split(".", 1)[1]
            words = tool.split("_")
            new_tool = "_".join(reversed(words)) if len(words) > 1 and rng.random() < 0.5 else tool
            if rng.random() < 0.3:
                new_tool = _camel(new_tool)
            self.tool_map[s.name] = f"{self.app_map[s.app]}.{new_tool}"
            camel = rng.random() < 0.4
            self.param_map[s.name] = {p: (_camel(p) if camel else p) for p in s.properties}

    def text(self, t: str) -> str:
        """Rewrite tool names mentioned in descriptions ("lấy từ contacts.search")."""
        for old in sorted(self.tool_map, key=len, reverse=True):
            t = t.replace(old, self.tool_map[old])
        return t

    def spec(self, s: ToolSpec) -> ToolSpec:
        if s.name == ASK_TOOL:
            return s
        pm = self.param_map[s.name]
        params = copy.deepcopy(s.parameters)
        props = {}
        for k, v in s.properties.items():
            v = dict(v)
            if v.get("description"):
                v["description"] = self.text(v["description"])
            props[pm[k]] = v
        params["properties"] = props
        params["required"] = [pm.get(k, k) for k in s.required]
        if s.one_of:
            params["x-one-of"] = [[pm.get(k, k) for k in g] for g in s.one_of]
        return replace(s, name=self.tool_map[s.name], app=self.app_map[s.app], parameters=params, description=self.text(s.description))

    def call(self, c: dict[str, Any]) -> dict[str, Any]:
        name, args = c["name"], dict(c.get("arguments") or {})
        if name == ASK_TOOL:
            target = args.get("tool_name", "")
            pm = self.param_map.get(target, {})
            args["tool_name"] = self.tool_map.get(target, target)
            args["params"] = [pm.get(p, p) for p in args.get("params") or []]
            return {"name": name, "arguments": args}
        pm = self.param_map.get(name, {})
        return {"name": self.tool_map.get(name, name), "arguments": {pm.get(k, k): v for k, v in args.items()}}


def rename_sample(sample: Sample, specs: list[ToolSpec], apps: dict[str, dict], seed: int) -> tuple[Sample, list[ToolSpec], dict]:
    """A renamed copy of ``sample``, its renamed tool specs and app metadata."""
    r = Renamer(specs, random.Random(f"{sample.id}:{seed}"))
    new_specs = [r.spec(s) for s in specs]
    new_apps = {r.app_map.get(a, a): meta for a, meta in apps.items()}
    history = []
    for turn in sample.history:
        t = dict(turn)
        if t.get("tool_calls"):
            t["tool_calls"] = [r.call(c) for c in t["tool_calls"]]
        history.append(t)
    label = dict(sample.label or {})
    label["tool_calls"] = [r.call(c) for c in label.get("tool_calls") or []]
    installed: dict[str, dict] = {}
    for s in new_specs:
        installed.setdefault(s.app, {"tools": []})["tools"].append(s.name)
    context = dict(sample.context)
    context["installed_apps"] = installed
    new = replace(sample, id=f"{sample.id}#r{seed}", context=context, history=history, label=label)
    return new, new_specs, new_apps
