"""Qwen3-ASR wrapper (``pip install qwen-asr``). Works with the base or a fine-tuned checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class QwenASR:
    def __init__(
        self,
        model: str,
        backend: str = "transformers",
        language: str | None = "Vietnamese",
        max_batch_size: int = 32,
        max_new_tokens: int = 256,
        gpu_memory_utilization: float = 0.25,
        device: str = "cuda:0",
    ):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.language = language
        if backend == "vllm":
            self.model: Any = Qwen3ASRModel.LLM(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
                max_inference_batch_size=max_batch_size,
                max_new_tokens=max_new_tokens,
            )
        else:
            self.model = Qwen3ASRModel.from_pretrained(
                model,
                dtype=torch.bfloat16,
                device_map=device,
                max_inference_batch_size=max_batch_size,
                max_new_tokens=max_new_tokens,
            )

    def transcribe(self, paths: list[Path]) -> list[str]:
        audio = [str(p) for p in paths]
        language = [self.language] * len(audio) if self.language else None
        results = self.model.transcribe(audio=audio, language=language)
        return [r.text for r in results]

    def close(self) -> None:
        """Free GPU memory before the LLM is loaded."""
        import gc

        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
