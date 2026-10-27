"""Shared test helpers. Data-dependent tests are skipped when the competition data is not available.

Point ``LIFEMATE_DATA`` at the folder that holds ``train/`` and ``public_test/`` (default: ``../train+public_test``
next to the repository).
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings
from functools import lru_cache
from pathlib import Path

# the organisers' zaic_tools opens files without closing them; keep the test output readable
warnings.filterwarnings("ignore", category=ResourceWarning)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = Path(os.environ.get("LIFEMATE_DATA", REPO.parent / "train+public_test"))
HAS_DATA = (DATA / "train" / "train.jsonl").exists()
requires_data = unittest.skipUnless(HAS_DATA, f"competition data not found at {DATA} (set LIFEMATE_DATA)")

# Training labels known to be wrong: gold sends a message to a chat id that no tool ever returned.
KNOWN_NOISY_LABELS = {"train_1696"}


@lru_cache(maxsize=None)
def split(name: str = "train"):
    from lifemate.data import Split

    return Split.resolve(DATA, name)


class FakeSession:
    """Minimal stand-in for a zaic_tools session."""

    def __init__(self, valid: bool = True):
        self.valid = valid
        self.calls: list[dict] = []

    def validate(self, name, arguments):
        return {"ok": self.valid, "error": None if self.valid else "thiếu tham số bắt buộc: x"}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        return {"ok": True}

    def get_tool_calls(self):
        return list(self.calls)

    def get_state_hash(self):
        return "0" * 64
