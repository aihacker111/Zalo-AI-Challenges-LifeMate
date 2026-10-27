#!/usr/bin/env python3
"""LifeMate ASR manifests -> lhotse cuts with 80-dim fbank, ready for icefall's zipformer/finetune.py.

Input: ``train.jsonl`` / ``val.jsonl`` written by ``lifemate asr-manifest`` (``{"audio", "text"}``).
Output in ``--out-dir``: ``cuts_train.jsonl.gz``, ``cuts_dev.jsonl.gz`` and their features.

* Text is converted to spoken form (lower case, numbers in words), the form gipformer writes.
  Training utterances whose spoken form is ambiguous ("RV245", "7h30") are dropped, as are those
  containing pieces the BPE model does not know. Dev keeps every utterance.
* Audio is resampled to 16 kHz. With ``--speed-perturb`` (default) train gets 0.9x and 1.1x copies.

Runs in the gipformer environment (``scripts/gipformer_setup.sh``)::

    python recipes/gipformer/prepare_cuts.py --manifest-dir outputs/asr_ft \
        --bpe-model models/gipformer/base/bpe.model --out-dir outputs/gipformer/data_ft
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_normalizer():
    """``src/lifemate/asr/normalize_vi.py`` loaded by path (the lifemate package is not installed here)."""
    path = REPO / "src" / "lifemate" / "asr" / "normalize_vi.py"
    spec = importlib.util.spec_from_file_location("normalize_vi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = row["text"]
                if "<asr_text>" in text:  # Qwen3-ASR format: "language Vietnamese<asr_text>..."
                    text = text.split("<asr_text>", 1)[1]
                rows.append({"audio": row["audio"], "text": text})
    return rows


def bpe_is_upper(sp) -> bool:
    """Whether the BPE vocabulary is upper case (icefall LibriSpeech style) rather than lower case."""
    upper = lower = 0
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        upper += sum(c.isupper() for c in piece)
        lower += sum(c.islower() for c in piece)
    return upper > lower


def build_cuts(rows: list[dict], name: str, sp, norm, strict: bool):
    from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet

    upper = bpe_is_upper(sp)
    unk = sp.unk_id()
    recordings, supervisions = [], []
    dropped: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for row in rows:
        audio = Path(row["audio"])
        if not audio.is_absolute():
            audio = (Path.cwd() / audio).resolve()
        text, unambiguous = norm.spoken_form(row["text"])
        if upper:
            text = text.upper()
        reason = None
        if not text:
            reason = "empty text"
        elif strict and not unambiguous:
            reason = "ambiguous reading"
        elif strict and unk in sp.encode(text):
            reason = "unknown BPE piece"
        if reason:
            dropped[reason] += 1
            examples.setdefault(reason, []).append(row["text"])
            continue
        rec_id = f"{audio.parent.name}-{audio.stem}"
        rec = Recording.from_file(audio, recording_id=rec_id)
        recordings.append(rec)
        supervisions.append(
            SupervisionSegment(
                id=rec_id, recording_id=rec_id, start=0.0, duration=rec.duration, channel=0, text=text, language="vi"
            )
        )
    for reason, n in dropped.items():
        logging.info("%s: dropped %d utterances (%s), e.g. %s", name, n, reason, examples[reason][:3])
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(recordings),
                                 supervisions=SupervisionSet.from_segments(supervisions))
    return cuts.resample(16000), dropped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", type=Path, required=True, help="directory with train.jsonl (and val.jsonl)")
    p.add_argument("--bpe-model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--speed-perturb", type=int, default=1, help="1: add 0.9x and 1.1x copies of train")
    p.add_argument("--num-jobs", type=int, default=8)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import sentencepiece as spm
    import torch
    from lhotse import Fbank, FbankConfig, LilcomChunkyWriter

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))
    norm = load_normalizer()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extractor = Fbank(FbankConfig(num_mel_bins=80))

    for name, manifest, strict in (("train", "train.jsonl", True), ("dev", "val.jsonl", False)):
        path = args.manifest_dir / manifest
        rows = read_manifest(path) if path.exists() else []
        if not rows:
            if name == "train":
                raise SystemExit(f"no utterances in {path}")
            # Full-data run: no held-out set. Validate on a slice of train (only to watch the loss).
            rows = read_manifest(args.manifest_dir / "train.jsonl")[:200]
            logging.info("dev: %s is empty, using the first %d train utterances", path, len(rows))
        cuts, _ = build_cuts(rows, name, sp, norm, strict)
        if name == "train" and args.speed_perturb:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=args.out_dir / f"feats_{name}",
            num_jobs=args.num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        out = args.out_dir / f"cuts_{name}.jsonl.gz"
        cuts.to_file(out)
        hours = sum(c.duration for c in cuts) / 3600
        over = sum(c.duration > 20.0 for c in cuts)
        logging.info("%s: %d cuts, %.2f h -> %s (%d longer than 20 s are skipped by finetune.py)",
                     name, len(cuts), hours, out, over)
        if name == "train":
            sample = next(iter(cuts))
            logging.info("example text: %s", sample.supervisions[0].text)


if __name__ == "__main__":
    main()
