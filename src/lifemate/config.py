"""Run-time configuration (dataclasses), loadable from YAML or JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class GuardConfig:
    validate_before_call: bool = True  # env.validate() first: it is not logged
    max_fix_attempts: int = 2  # schema errors returned to the model before the call is dropped
    block_duplicate_reads: bool = True  # gold never repeats an identical read
    block_duplicate_writes: bool = False  # gold repeats identical writes (e.g. redeem a reward twice)
    require_seen_refs: bool = True  # x-ref ids must come from a tool result or the history
    drop_unknown_args: bool = True
    drop_empty_optional_args: bool = True
    coerce_types: bool = True


@dataclass
class PromptConfig:
    days_ahead: int = 8
    fewshot_k: int = 3
    result_max_items: int = 15
    result_max_chars: int = 1500
    history_result_max_chars: int = 600


@dataclass
class AgentConfig:
    max_steps: int = 8
    time_budget_s: float = 25.0
    temperature: float = 0.0
    max_new_tokens: int = 384
    constrained: bool = True
    guard: GuardConfig = field(default_factory=GuardConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else None  # type: ignore[misc]
        kwargs[f.name] = _build(type(default), value) if is_dataclass(default) and isinstance(value, dict) else value
    return cls(**kwargs)


def load_agent_config(path: str | Path | None) -> AgentConfig:
    if path is None:
        return AgentConfig()
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        import yaml  # optional dependency: pip install lifemate[yaml]

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return _build(AgentConfig, data.get("agent", data))


def to_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)
