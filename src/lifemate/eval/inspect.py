"""Turn per-sample scores into something actionable: which arguments differ, and on which samples."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from lifemate.eval.report import classify


def load_report(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _short(v: Any, n: int = 60) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def argument_diffs(pred: list[dict[str, Any]], gold: list[dict[str, Any]]) -> list[tuple[str, str, Any, Any]]:
    """(tool, argument, predicted, gold) for unmatched calls paired by tool name, in order."""
    out = []
    remaining = list(pred)
    for g in gold:
        p = next((c for c in remaining if c.get("name") == g.get("name")), None)
        if p is None:
            continue
        remaining.remove(p)
        pa, ga = p.get("arguments") or {}, g.get("arguments") or {}
        for k in sorted(set(pa) | set(ga)):
            if pa.get(k, "<thiếu>") != ga.get(k, "<thiếu>"):
                out.append((g["name"], k, pa.get(k, "<thiếu>"), ga.get(k, "<thiếu>")))
    return out


def summarize_report(rows: list[dict[str, Any]], kinds: dict[str, str]) -> str:
    """Counts of the most frequent missing / extra calls and argument differences."""
    missing: Counter = Counter()
    extra: Counter = Counter()
    args: Counter = Counter()
    for r in rows:
        missing.update(c["name"] for c in r["unmatched_gold"])
        extra.update(c["name"] for c in r["unmatched_pred"])
        for tool, k, *_ in argument_diffs(r["unmatched_pred"], r["unmatched_gold"]):
            args[f"{tool}.{k}"] += 1
    lines = ["Call gold mà model không gọi (hoặc gọi sai tham số):"]
    lines += [f"  {n:>4}  {name} [{kinds.get(name, '?')}]" for name, n in missing.most_common(15)]
    lines += ["", "Call model gọi thừa (hoặc sai tham số):"]
    lines += [f"  {n:>4}  {name} [{kinds.get(name, '?')}]" for name, n in extra.most_common(15)]
    lines += ["", "Tham số lệch nhiều nhất (cùng tool, khác giá trị hoặc thiếu/thừa):"]
    lines += [f"  {n:>4}  {key}" for key, n in args.most_common(20)]
    return "\n".join(lines)


def show_samples(rows: list[dict[str, Any]], kinds: dict[str, str], texts: dict[str, str], label: str | None, limit: int) -> str:
    blocks = []
    for r in rows:
        if r["score"] == 1.0:
            continue

        class _S:  # adapter: classify() expects attribute access
            pass

        s = _S()
        s.__dict__.update(r)
        labels = classify(s, kinds)  # type: ignore[arg-type]
        if label and label not in labels:
            continue
        lines = [f"── {r['id']}  score={r['score']:.2f}  hash_ok={r['hash_ok']}  [{', '.join(labels)}]"]
        if r["id"] in texts:
            lines.append(f'   "{texts[r["id"]]}"')
        for c in r["unmatched_gold"]:
            lines.append(f"   - gold  {c['name']} {_short(c.get('arguments'), 110)}")
        for c in r["unmatched_pred"]:
            lines.append(f"   + model {c['name']} {_short(c.get('arguments'), 110)}")
        for tool, k, p, g in argument_diffs(r["unmatched_pred"], r["unmatched_gold"]):
            lines.append(f"     ≠ {tool}.{k}: model={_short(p)} gold={_short(g)}")
        blocks.append("\n".join(lines))
        if len(blocks) >= limit:
            break
    return "\n\n".join(blocks)
