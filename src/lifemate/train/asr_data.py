"""ASR fine-tuning manifests from the training split (every user turn: history and final request)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lifemate.data import Split


def build_manifests(
    split: Split,
    out_dir: str | Path,
    val_ratio: float = 0.05,
    language: str = "Vietnamese",
    val_ids_path: str | Path | None = None,
) -> tuple[int, int]:
    """Writes ``train.jsonl`` and ``val.jsonl`` in the format of Qwen3-ASR's ``finetuning/qwen3_asr_sft.py``:
    ``{"audio": <wav path>, "text": "language Vietnamese<asr_text><transcript>"}``, split by sample id."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = [0, 0]
    val_ids = None
    if val_ids_path:
        val_ids = {x.strip() for x in Path(val_ids_path).read_text(encoding="utf-8").splitlines() if x.strip()}
    with open(out / "train.jsonl", "w", encoding="utf-8") as ftr, open(out / "val.jsonl", "w", encoding="utf-8") as fva:
        for s in split:
            if val_ids is not None:
                is_val = s.id in val_ids
            else:
                is_val = int(hashlib.md5(s.id.encode()).hexdigest(), 16) % 1000 < val_ratio * 1000
            f = fva if is_val else ftr
            for rel, text in s.user_turn_texts().items():
                row = {"audio": str(split.audio_path(rel)), "text": f"language {language}<asr_text>{text}"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[is_val] += 1
    return counts[0], counts[1]
