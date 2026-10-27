"""Error taxonomy on top of :class:`~lifemate.eval.scorer.SampleScore`."""

from __future__ import annotations

from collections import Counter
from typing import Any

from lifemate.eval.scorer import SampleScore


def classify(s: SampleScore, kinds: dict[str, str]) -> list[str]:
    """Labels explaining why a sample lost points (empty for a perfect sample)."""
    if s.score == 1.0:
        return []
    labels: list[str] = []
    if s.gold_empty:
        return ["oos_false_action"]
    if s.n_pred == 0:
        return ["oos_false_empty"]
    if not s.hash_ok:
        labels.append("wrong_final_state")
    if s.schema_errors:
        labels.append("schema_error")

    def kind(c: dict[str, Any]) -> str:
        return kinds.get(c.get("name", ""), "write")

    extra = Counter(kind(c) for c in s.unmatched_pred)
    missing = Counter(kind(c) for c in s.unmatched_gold)
    if extra["read"]:
        labels.append("extra_read")
    if missing["read"]:
        labels.append("missing_read")
    if extra["write"] or missing["write"]:
        labels.append("write_mismatch")
    if extra["meta"] or missing["meta"]:
        labels.append("ask_mismatch")
    return labels or ["other"]


def taxonomy(scores: list[SampleScore], kinds: dict[str, str]) -> Counter:
    c: Counter = Counter()
    for s in scores:
        for label in classify(s, kinds):
            c[label] += 1
    return c


def format_report(summary: dict[str, float], tax: Counter, n: int) -> str:
    lines = [
        f"score         {summary['score']:.2f}",
        f"samples       {summary['samples']}",
        f"perfect       {100 * summary['perfect']:.1f}%",
        f"hash ok       {100 * summary['hash_ok']:.1f}%",
        f"schema errors {summary['schema_errors']}",
        "",
        "errors (samples, a sample can have several):",
    ]
    for label, count in tax.most_common():
        lines.append(f"  {label:<20} {count:>5}  ({100 * count / max(n, 1):.1f}%)")
    return "\n".join(lines)
