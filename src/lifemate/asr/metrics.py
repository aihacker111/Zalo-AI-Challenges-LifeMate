"""ASR metrics that matter for this task: WER plus error rates on numbers and on reference words."""

from __future__ import annotations

import re

from lifemate.text import fold, tokens

_NUM = re.compile(r"\d+")


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(tokens(r), tokens(h)) for r, h in zip(refs, hyps))
    words = sum(len(tokens(r)) for r in refs)
    return errors / max(words, 1)


def wer_no_tone(refs: list[str], hyps: list[str]) -> float:
    """WER after removing diacritics: separates tone errors from word errors."""
    return wer([fold(r) for r in refs], [fold(h) for h in hyps])


def number_error_rate(refs: list[str], hyps: list[str]) -> float:
    """Share of utterances whose digit sequences differ (amounts, times, phone numbers)."""
    with_num = [(r, h) for r, h in zip(refs, hyps) if _NUM.search(r)]
    bad = sum(_NUM.findall(r) != _NUM.findall(h) for r, h in with_num)
    return bad / max(len(with_num), 1)
