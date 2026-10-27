"""Qwen3-ASR wrappers.

* :class:`QwenASRvLLM` (default): runs Qwen3-ASR with vLLM, which supports it natively. Needs nothing beyond the
  vLLM already used for the LLM, and works with transformers 5.
* :class:`QwenASRPackage`: the ``qwen-asr`` package (``pip install qwen-asr``). Its bundled model code targets
  transformers 4.x and fails to import with transformers 5.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

ASR_TAG = "<asr_text>"


def parse_asr_output(text: str) -> str:
    """Qwen3-ASR answers ``language Vietnamese<asr_text>transcript``; keep the transcript."""
    if ASR_TAG in text:
        text = text.split(ASR_TAG, 1)[1]
    return text.replace("<|im_end|>", "").strip()


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


class QwenASRvLLM:
    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.4,
        max_new_tokens: int = 256,
        max_model_len: int = 4096,
        batch_size: int = 64,
        seed: int = 0,
    ):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            allowed_local_media_path="/",
            limit_mm_per_prompt={"audio": 1},
            seed=seed,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed)
        self.batch_size = batch_size

    def transcribe(self, paths: list[Path]) -> list[str]:
        conversations = [
            [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": Path(p).resolve().as_uri()}}]}]
            for p in paths
        ]
        outputs = self.llm.chat(conversations, self.sampling, use_tqdm=False)
        return [parse_asr_output(o.outputs[0].text) for o in outputs]

    def close(self) -> None:
        del self.llm
        _free_gpu()


class QwenASRPackage:
    def __init__(self, model: str, language: str | None = "Vietnamese", max_batch_size: int = 32, max_new_tokens: int = 256, device: str = "cuda:0"):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.language = language
        self.model: Any = Qwen3ASRModel.from_pretrained(
            model, dtype=torch.bfloat16, device_map=device, max_inference_batch_size=max_batch_size, max_new_tokens=max_new_tokens
        )

    def transcribe(self, paths: list[Path]) -> list[str]:
        audio = [str(p) for p in paths]
        language = [self.language] * len(audio) if self.language else None
        return [r.text for r in self.model.transcribe(audio=audio, language=language)]

    def close(self) -> None:
        del self.model
        _free_gpu()


def QwenASR(model: str, backend: str = "vllm", **kwargs: Any) -> QwenASRvLLM | QwenASRPackage:
    """Factory used by the CLI: ``backend`` is ``vllm`` (default) or ``qwen-asr``."""
    if backend in ("qwen-asr", "transformers"):
        return QwenASRPackage(model, **kwargs)
    return QwenASRvLLM(model, **kwargs)
