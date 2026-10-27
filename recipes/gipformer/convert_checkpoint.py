#!/usr/bin/env python3
"""Turn gipformer's released ``model.pt`` into a checkpoint icefall's finetune.py can load.

finetune.py expects ``{"model": state_dict}`` and loads it strictly. The released file may be a bare
state dict, so the model is rebuilt with icefall's default Zipformer arguments (as gipformer's
``infer_pytorch.py`` does), the weights are loaded and checked, and ``{"model": ...}`` is saved.

    python recipes/gipformer/convert_checkpoint.py --src models/gipformer/base/model.pt \
        --bpe-model models/gipformer/base/bpe.model --out models/gipformer/base/pretrained.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ICEFALL = Path(os.environ.get("ICEFALL_DIR", Path(__file__).resolve().parents[2] / "third_party" / "icefall"))
sys.path[:0] = [str(ICEFALL / "egs" / "librispeech" / "ASR" / "zipformer"), str(ICEFALL)]


def main() -> None:
    import sentencepiece as spm
    import torch
    from train import add_model_arguments, get_model, get_params

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--bpe-model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context-size", type=int, default=2)
    add_model_arguments(p)
    args = p.parse_args()

    params = get_params()
    params.update(vars(args))
    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    model = get_model(params)
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M, vocab: {params.vocab_size}")
    print(f"missing keys: {len(missing)} {list(missing)[:5]}")
    print(f"unexpected keys: {len(unexpected)} {list(unexpected)[:5]}")
    if len(missing) > 0.05 * len(model.state_dict()):
        raise SystemExit("too many missing keys: the model arguments do not match this checkpoint")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
