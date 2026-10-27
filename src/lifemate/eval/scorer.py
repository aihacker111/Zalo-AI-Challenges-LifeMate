"""Offline re-implementation of the LifeMate metric.

Per sample (gold calls G, submitted calls S):

* gold ``[]``: 1 if the submission is ``[]`` too, else 0;
* otherwise 0 when the replayed state hash differs from the gold one or no call matches (M = 0), else
  ``F1 = 2PR / (P + R)`` with ``P = M / (M + U)``, ``R = M / |G|``.

A submitted call matches a gold call when the tool name is equal, every non-fuzzy argument is identical and every fuzzy
argument (``x-match: fuzzy`` in the catalog) has token-F1 > 0.6 after lower case + NFC. A call matches at most one
call and order is ignored, so M is a maximum bipartite matching. ``U`` counts every unmatched submitted call, including
calls rejected by the schema check. The final score is ``100 * mean(score)``.

The state hash of a submission is recomputed by replaying its calls with the split's own ``zaic_tools`` (that is what
the organisers do); the gold hash comes from replaying ``label.tool_calls``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lifemate.data import Split
from lifemate.text import norm, token_f1

FUZZY_THRESHOLD = 0.6


@dataclass
class SampleScore:
    id: str
    score: float
    hash_ok: bool
    matched: int
    n_gold: int
    n_pred: int
    schema_errors: int = 0
    gold_empty: bool = False
    unmatched_pred: list[dict[str, Any]] = field(default_factory=list)
    unmatched_gold: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unmatched(self) -> int:
        return self.n_pred - self.matched


def _same_value(a: Any, b: Any, lenient: bool) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if lenient:
        if isinstance(a, str) and isinstance(b, str):
            return norm(a) == norm(b)
        if isinstance(a, list) and isinstance(b, list) and all(not isinstance(x, (dict, list)) for x in a + b):
            return sorted(map(str, a)) == sorted(map(str, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
    return type(a) is type(b) and a == b


def calls_match(pred: dict[str, Any], gold: dict[str, Any], fuzzy: set[str], lenient: bool = False) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    for k, gv in ga.items():
        pv = pa[k]
        if k in fuzzy:
            if not (isinstance(pv, str) and isinstance(gv, str) and token_f1(pv, gv) > FUZZY_THRESHOLD):
                return False
        elif not _same_value(pv, gv, lenient):
            return False
    return True


def match_calls(
    pred: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    fuzzy_of: dict[str, set[str]],
    lenient: bool = False,
) -> list[tuple[int, int]]:
    """Maximum one-to-one matching (Kuhn's algorithm); returns (pred_index, gold_index) pairs."""
    adj = [[j for j, g in enumerate(gold) if calls_match(p, g, fuzzy_of.get(g.get("name", ""), set()), lenient)] for p in pred]
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(pred)):
        augment(i, set())
    return sorted((i, j) for j, i in owner.items())


def f1_score(matched: int, n_pred: int, n_gold: int) -> float:
    if matched == 0:
        return 0.0
    p, r = matched / n_pred, matched / n_gold
    return 2 * p * r / (p + r)


class Scorer:
    """Scores predictions against a labelled split (train, or any split with ``label``)."""

    def __init__(self, split: Split, lenient_args: bool = False):
        if not split.has_labels:
            raise ValueError(f"{split.name} has no labels")
        self.split = split
        self.lenient_args = lenient_args
        self.fuzzy_of = {
            name: {k for k, p in (t.get("parameters") or {}).get("properties", {}).items() if p.get("x-match") == "fuzzy"}
            for name, t in split.tools.items()
        }
        self._gold_hash: dict[str, str] = {}

    def replay(self, sample_id: str, calls: list[dict[str, Any]]) -> tuple[str, int]:
        """(state hash, number of schema-rejected calls) after replaying ``calls`` from the initial state."""
        session = self.split.open_session(sample_id)
        rejected = 0
        for c in calls:
            r = session.call(c.get("name", ""), c.get("arguments") or {})
            rejected += int(r.get("ok") is False)
        return session.get_state_hash(), rejected

    def gold_hash(self, sample_id: str) -> str:
        if sample_id not in self._gold_hash:
            self._gold_hash[sample_id] = self.replay(sample_id, self.split.get(sample_id).gold_calls or [])[0]
        return self._gold_hash[sample_id]

    def score_sample(self, sample_id: str, pred_calls: list[dict[str, Any]]) -> SampleScore:
        gold = self.split.get(sample_id).gold_calls or []
        pred = list(pred_calls or [])
        pred_hash, rejected = self.replay(sample_id, pred)
        hash_ok = pred_hash == self.gold_hash(sample_id)
        pairs = match_calls(pred, gold, self.fuzzy_of, self.lenient_args)
        mp, mg = {i for i, _ in pairs}, {j for _, j in pairs}
        if not gold:
            score = 1.0 if not pred else 0.0
        else:
            score = f1_score(len(pairs), len(pred), len(gold)) if hash_ok else 0.0
        return SampleScore(
            id=sample_id,
            score=score,
            hash_ok=hash_ok,
            matched=len(pairs),
            n_gold=len(gold),
            n_pred=len(pred),
            schema_errors=rejected,
            gold_empty=not gold,
            unmatched_pred=[c for i, c in enumerate(pred) if i not in mp],
            unmatched_gold=[c for j, c in enumerate(gold) if j not in mg],
        )

    def score_predictions(self, predictions: dict[str, list[dict[str, Any]]], ids: Iterable[str] | None = None) -> list[SampleScore]:
        """Missing predictions count as ``[]``."""
        ids = list(ids) if ids is not None else [s.id for s in self.split]
        return [self.score_sample(i, predictions.get(i, [])) for i in ids]


def summarize(scores: list[SampleScore]) -> dict[str, float]:
    n = len(scores) or 1
    return {
        "score": 100 * sum(s.score for s in scores) / n,
        "samples": len(scores),
        "hash_ok": sum(s.hash_ok for s in scores) / n,
        "perfect": sum(s.score == 1.0 for s in scores) / n,
        "schema_errors": sum(s.schema_errors for s in scores),
    }


def load_submission(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row.get("tool_calls") or []
    return out


def dump_scores(scores: list[SampleScore], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
