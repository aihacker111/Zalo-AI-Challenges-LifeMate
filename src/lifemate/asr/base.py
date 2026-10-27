"""Transcripts keyed by the audio path relative to the split folder (``audio/<id>/<id>_u1.wav``)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from lifemate.data import Sample, Split
from lifemate.text import nfc


class Transcriber(Protocol):
    def transcribe(self, paths: list[Path]) -> list[str]: ...


class TranscriptStore(dict):
    """``{audio_rel_path: text}`` persisted as JSON Lines ``{"audio": ..., "text": ...}``."""

    @classmethod
    def load(cls, path: str | Path) -> TranscriptStore:
        store = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    store[row["audio"]] = row["text"]
        return store

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for audio, text in sorted(self.items()):
                f.write(json.dumps({"audio": audio, "text": text}, ensure_ascii=False) + "\n")


def oracle_transcripts(samples: Iterable[Sample]) -> TranscriptStore:
    """Gold transcripts of a labelled split (upper bound of the ASR stage)."""
    store = TranscriptStore()
    for s in samples:
        store.update(s.user_turn_texts())
    return store


def transcribe_split(split: Split, transcriber: Transcriber, batch_size: int = 32, store: TranscriptStore | None = None) -> TranscriptStore:
    """Transcribe every user turn of a split (history turns included), skipping paths already in ``store``."""
    store = store if store is not None else TranscriptStore()
    todo = sorted({p for s in split for p in s.user_turn_audio()} - set(store))
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        texts = transcriber.transcribe([split.audio_path(p) for p in chunk])
        store.update({p: nfc(t).strip() for p, t in zip(chunk, texts)})
    return store
