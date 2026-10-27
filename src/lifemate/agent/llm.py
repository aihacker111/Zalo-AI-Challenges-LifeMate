"""Decoding backends.

* :class:`VLLMBackend` — offline vLLM in the inference container (greedy, prefix caching, structured outputs).
* :class:`OpenAICompatBackend` — any OpenAI-compatible server (``vllm serve`` during development, or a large teacher
  model to generate training data; never used at inference, which must run offline).
* :class:`OracleBackend` / :class:`ScriptedBackend` — replay known actions; used by tests and the pipeline sanity check.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from typing import Any, Protocol

from lifemate.agent.render import format_action
from lifemate.catalog import FINISH
from lifemate.data import Sample


class Backend(Protocol):
    def generate(self, messages: list[dict[str, str]], schema: dict[str, Any] | None, temperature: float, max_tokens: int) -> str: ...


class VLLMBackend:
    def __init__(
        self,
        model: str,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.7,
        seed: int = 0,
        enable_thinking: bool = False,
        lora_path: str | None = None,
        **llm_kwargs: Any,
    ):
        from vllm import LLM  # imported lazily: vLLM is only needed on a GPU machine

        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=seed,
            enable_lora=lora_path is not None,
            **llm_kwargs,
        )
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.lora = None
        if lora_path:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("adapter", 1, lora_path)

    def _sampling(self, schema: dict[str, Any] | None, temperature: float, max_tokens: int) -> Any:
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "seed": self.seed}
        if schema is None:
            return SamplingParams(**kwargs)
        try:  # vLLM >= 0.10
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema), **kwargs)
        except ImportError:  # older vLLM
            from vllm.sampling_params import GuidedDecodingParams

            return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **kwargs)

    def generate(self, messages, schema, temperature, max_tokens):
        out = self.llm.chat(
            messages,
            self._sampling(schema, temperature, max_tokens),
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            lora_request=self.lora,
            use_tqdm=False,
        )
        return out[0].outputs[0].text


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 60.0, vllm_extras: bool = True, seed: int = 0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.vllm_extras = vllm_extras
        self.seed = seed

    def generate(self, messages, schema, temperature, max_tokens):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.seed,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "schema": schema}}
        if self.vllm_extras:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - URL comes from the user's config
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"] or ""


class ScriptedBackend:
    """Returns the given outputs in order, then ``finish``."""

    def __init__(self, outputs: Iterable[str]):
        self.outputs = list(outputs)

    def generate(self, messages, schema, temperature, max_tokens):
        return self.outputs.pop(0) if self.outputs else format_action(FINISH)


class OracleBackend:
    """Emits a labelled sample's gold calls, then ``finish``. Proves the pipeline loses no points by itself."""

    def __init__(self) -> None:
        self._queue: list[str] = []

    def begin(self, sample: Sample) -> None:
        self._queue = [format_action(c["name"], c.get("arguments")) for c in sample.gold_calls or []]

    def generate(self, messages, schema, temperature, max_tokens):
        return self._queue.pop(0) if self._queue else format_action(FINISH)
