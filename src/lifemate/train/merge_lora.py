"""Merge a LoRA adapter into its base model (for a self-contained inference image)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    import transformers
    from peft import PeftModel

    from lifemate.train.sft_lora import SFTConfig, load_model

    model = load_model(SFTConfig(model=args.base, attn_implementation=None))
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model written to {args.out}")


if __name__ == "__main__":
    main()
