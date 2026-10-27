"""Validation splits that do not leak paraphrases of the same scenario, and leave-apps-out splits."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from lifemate.data import Sample


def scenario_key(sample: Sample) -> str:
    """Samples with the same sequence of gold tool names are paraphrases of one scenario."""
    calls = sample.gold_calls or []
    return "|".join(c["name"] for c in calls) or "<empty>"


def _bucket(key: str, k: int, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) % k


def group_kfold(samples: Iterable[Sample], k: int = 5, seed: int = 0) -> dict[str, int]:
    """sample id -> fold, grouping by :func:`scenario_key`."""
    return {s.id: _bucket(scenario_key(s), k, seed) for s in samples}


def apps_used(sample: Sample) -> set[str]:
    return {c["name"].split(".", 1)[0] for c in sample.gold_calls or []} - {"sys"}


def leave_apps_out(samples: list[Sample], held_out: set[str]) -> tuple[list[str], list[str]]:
    """(train ids, val ids): a sample goes to val when its gold calls use a held-out app.

    Training samples may still list a held-out app among the installed tools (its schema is visible, its use is not).
    """
    train, val = [], []
    for s in samples:
        (val if apps_used(s) & held_out else train).append(s.id)
    return train, val


def pick_held_out_apps(samples: list[Sample], n: int = 10, seed: int = 0) -> set[str]:
    """Deterministic choice of ``n`` mid-frequency apps (the most common ones are kept for training)."""
    freq = Counter(a for s in samples for a in apps_used(s))
    ranked = sorted(freq, key=lambda a: (-freq[a], a))
    pool = ranked[len(ranked) // 5 :]  # skip the top 20%
    pool.sort(key=lambda a: hashlib.md5(f"{seed}:{a}".encode()).hexdigest())
    return set(pool[:n])
