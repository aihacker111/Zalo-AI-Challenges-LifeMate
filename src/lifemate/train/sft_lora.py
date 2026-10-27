"""LoRA supervised fine-tuning that fits one 24 GB GPU (RTX 3090/4090, A30).

Usage::

    python -m lifemate.train.sft_lora --config configs/sft_lora_24gb.yaml

Only assistant turns (the action JSON and the end-of-turn token) carry loss. Spans are located on the rendered chat
text and mapped to tokens with the tokenizer's offsets, so this works whatever the chat template does with empty
``<think>`` blocks.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SFTConfig:
    model: str = "Qwen/Qwen3.5-4B"
    train_file: str = "outputs/sft/train.jsonl"
    eval_file: str | None = "outputs/sft/val.jsonl"
    output_dir: str = "checkpoints/sft-lora"
    max_length: int = 12288
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: Any = "all-linear"
    learning_rate: float = 1e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    seed: int = 42
    enable_thinking: bool = False
    attn_implementation: str | None = "sdpa"
    extra_trainer_args: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | None, overrides: list[str]) -> SFTConfig:
    data: dict[str, Any] = {}
    if path:
        text = Path(path).read_text(encoding="utf-8")
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    for item in overrides:
        k, v = item.split("=", 1)
        try:
            data[k] = json.loads(v)
        except json.JSONDecodeError:
            data[k] = v
    return SFTConfig(**data)


def assistant_char_spans(text: str, messages: list[dict[str, str]], end_token: str) -> list[tuple[int, int]]:
    """Character spans of each assistant message content plus its end-of-turn token, in order."""
    spans, pos = [], 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        start = text.find(m["content"], pos)
        if start < 0:
            raise ValueError("assistant content not found in rendered chat")
        end = start + len(m["content"])
        eot = text.find(end_token, end)
        end = eot + len(end_token) if eot >= 0 else end
        spans.append((start, end))
        pos = end
    return spans


def encode(tokenizer: Any, messages: list[dict[str, str]], max_length: int, enable_thinking: bool) -> dict[str, list[int]] | None:
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=enable_thinking)
    end_token = "<|im_end|>" if "<|im_end|>" in text else (tokenizer.eos_token or "")
    spans = assistant_char_spans(text, messages, end_token)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_length:
        return None
    labels = [-100] * len(ids)
    k = 0
    for i, (a, b) in enumerate(offsets):
        while k < len(spans) and a >= spans[k][1]:
            k += 1
        if k < len(spans) and b > spans[k][0] and a < spans[k][1]:
            labels[i] = ids[i]
    return {"input_ids": ids, "labels": labels}


class JsonlChatDataset:
    def __init__(self, path: str, tokenizer: Any, cfg: SFTConfig):
        self.rows: list[dict[str, list[int]]] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                enc = encode(tokenizer, json.loads(line)["messages"], cfg.max_length, cfg.enable_thinking)
                if enc is None:
                    skipped += 1
                else:
                    self.rows.append(enc)
        print(f"{path}: {len(self.rows)} examples, {skipped} longer than {cfg.max_length} tokens skipped")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.rows[i]


def collate(pad_id: int):
    import torch

    def fn(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    return fn


def load_model(cfg: SFTConfig) -> Any:
    import torch
    import transformers

    kwargs = {"torch_dtype": torch.bfloat16}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    except (ValueError, KeyError):  # multimodal checkpoints (e.g. Qwen3.5) register another auto class
        return transformers.AutoModelForImageTextToText.from_pretrained(cfg.model, **kwargs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. learning_rate=5e-5")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    random.seed(cfg.seed)
    transformers.set_seed(cfg.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=cfg.target_modules, task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()

    train_ds = JsonlChatDataset(cfg.train_file, tokenizer, cfg)
    eval_ds = JsonlChatDataset(cfg.eval_file, tokenizer, cfg) if cfg.eval_file and Path(cfg.eval_file).exists() else None
    targs = transformers.TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # enabled on the model above
        group_by_length=True,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        **cfg.extra_trainer_args,
    )
    trainer = transformers.Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
