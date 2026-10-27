"""Text helpers shared by the scorer, the guard and retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WORD = re.compile(r"[^\W_]+")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm(s: str) -> str:
    """NFC, lower case, collapsed whitespace."""
    return " ".join(nfc(str(s)).lower().split())


def fold(s: str) -> str:
    """Lower case without Vietnamese diacritics ("Hương" -> "huong"), as zaic_tools searches do."""
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def tokens(s: str) -> list[str]:
    """Word tokens after lower case + NFC (the scorer's normalisation for fuzzy arguments)."""
    return _WORD.findall(norm(s))


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 (SQuAD style)."""
    p, g = tokens(pred), tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)
