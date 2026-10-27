"""Vietnamese-specific ASR engines, much stronger than Qwen3-ASR-0.6B on regional accents.

* :class:`TransducerASR` — Zipformer RNN-T models exported for sherpa-onnx, e.g. gipformer
  (``g-group-ai-lab/gipformer1.5-68M-rnnt``, 68M parameters, MIT). Runs on CPU.
  ``pip install sherpa-onnx sherpa-onnx-core soundfile huggingface_hub``
* :class:`ChunkFormerASR` — ChunkFormer CTC/RNN-T (``khanhld/chunkformer-ctc-large-vie``, 110M). Runs on GPU.
  ``pip install chunkformer``

Usage mirrors the official scripts (gipformer ``infer_onnx.py``; ChunkFormer ``batch_decode``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

GIPFORMER = "g-group-ai-lab/gipformer1.5-68M-rnnt"
CHUNKFORMER = "khanhld/chunkformer-ctc-large-vie"


def _resolve_dir(model: str, patterns: list[str]) -> Path:
    """A local directory, or a Hugging Face repo id downloaded once into the HF cache."""
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=patterns))


class TransducerASR:
    """sherpa-onnx offline transducer (encoder/decoder/joiner ONNX + tokens.txt)."""

    def __init__(
        self,
        model: str = GIPFORMER,
        quantize: str = "fp32",
        num_threads: int | None = None,
        decoding_method: str = "modified_beam_search",
        batch_size: int = 32,
        provider: str = "cpu",
    ):
        import sherpa_onnx

        suffix = ".int8.onnx" if quantize == "int8" else ".onnx"
        d = _resolve_dir(model, ["*.onnx", "tokens.txt"])
        files = {part: d / f"{part}{suffix}" for part in ("encoder", "decoder", "joiner")}
        missing = [str(f) for f in [*files.values(), d / "tokens.txt"] if not f.exists()]
        if missing:
            raise FileNotFoundError(f"missing model files: {missing}")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            tokens=str(d / "tokens.txt"),
            num_threads=num_threads or min(16, os.cpu_count() or 4),
            sample_rate=16000,
            feature_dim=80,
            decoding_method=decoding_method,
            provider=provider,
        )
        self.batch_size = batch_size

    def transcribe(self, paths: list[Path]) -> list[str]:
        import soundfile as sf

        texts: list[str] = []
        for i in range(0, len(paths), self.batch_size):
            streams = []
            for p in paths[i : i + self.batch_size]:
                samples, sr = sf.read(str(p), dtype="float32")
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                stream = self.recognizer.create_stream()
                stream.accept_waveform(sr, samples)
                streams.append(stream)
            self.recognizer.decode_streams(streams)
            texts.extend(s.result.text.strip() for s in streams)
        return texts

    def close(self) -> None:
        del self.recognizer


class ChunkFormerASR:
    def __init__(self, model: str = CHUNKFORMER, device: str = "cuda:0", total_batch_duration: int = 1800):
        import torch
        from chunkformer import ChunkFormerModel

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model: Any = ChunkFormerModel.from_pretrained(model).to(self.device).eval()
        self.total_batch_duration = total_batch_duration

    def transcribe(self, paths: list[Path]) -> list[str]:
        import torch

        with torch.no_grad():
            out = self.model.batch_decode(
                audio_paths=[str(p) for p in paths],
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=self.total_batch_duration,
            )
        return [str(t).strip() for t in out]

    def close(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
