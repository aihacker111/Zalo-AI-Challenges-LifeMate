#!/usr/bin/env python3
"""icefall's ``egs/librispeech/ASR/zipformer/finetune.py`` on LifeMate cuts.

finetune.py reads its data through ``LibriSpeechAsrDataModule`` (GigaSpeech train, LibriSpeech dev).
This wrapper swaps in a data module that returns ``cuts_train.jsonl.gz`` / ``cuts_dev.jsonl.gz`` from
``--manifest-dir`` (made by ``prepare_cuts.py``), validates more often than the LibriSpeech default,
then runs finetune.py unchanged. Every finetune.py option works, e.g.::

    python recipes/gipformer/train_gipformer.py --manifest-dir outputs/gipformer/data_ft \
        --bpe-model models/gipformer/base/bpe.model --finetune-ckpt models/gipformer/base/pretrained.pt \
        --exp-dir checkpoints/gipformer-ft --num-epochs 10 --base-lr 0.002 --max-duration 600 \
        --use-fp16 1 --enable-musan 0 --world-size 1

Environment: ``ICEFALL_DIR`` (default ``third_party/icefall``), ``VALID_INTERVAL`` (batches, default 100).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ICEFALL = Path(os.environ.get("ICEFALL_DIR", Path(__file__).resolve().parents[2] / "third_party" / "icefall"))
sys.path[:0] = [str(ICEFALL / "egs" / "librispeech" / "ASR" / "zipformer"), str(ICEFALL)]

import asr_datamodule  # noqa: E402  (icefall recipe module)
import finetune  # noqa: E402  (icefall recipe module)
from lhotse import CutSet, load_manifest, load_manifest_lazy  # noqa: E402


class LifeMateAsrDataModule(asr_datamodule.LibriSpeechAsrDataModule):
    """The data finetune.py asks for, mapped to the LifeMate cuts."""

    def gigaspeech_subset_small_cuts(self) -> CutSet:  # training data
        return load_manifest_lazy(self.args.manifest_dir / "cuts_train.jsonl.gz")

    def dev_clean_cuts(self) -> CutSet:  # validation set ("librispeech" in the log)
        return load_manifest(self.args.manifest_dir / "cuts_dev.jsonl.gz")

    def dev_other_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    def gigaspeech_dev_cuts(self) -> CutSet:  # second validation set: a few dev cuts, to keep it cheap
        return self.dev_clean_cuts().subset(first=32)


_get_params = finetune.get_params


def get_params():
    params = _get_params()
    params.valid_interval = int(os.environ.get("VALID_INTERVAL", 100))
    params.log_interval = int(os.environ.get("LOG_INTERVAL", 20))
    return params


finetune.LibriSpeechAsrDataModule = LifeMateAsrDataModule
finetune.get_params = get_params

if __name__ == "__main__":
    finetune.main()
