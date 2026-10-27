"""Gold trajectories rendered exactly as the agent sees them at inference time.

Each example is ``{"id", "messages"}``: system + user prompt, then for every gold call an assistant message (the
action JSON) followed by a user message (the tool result), and a final ``finish`` unless the last call is
``sys.ask_missing_info`` (the agent stops there). ``tests/test_sft_parity.py`` checks this equals what the agent loop
produces when it replays the gold calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lifemate.agent.fewshot import Retriever, render_examples
from lifemate.agent.prompt import PromptBuilder
from lifemate.agent.render import format_action, format_observation
from lifemate.catalog import ASK_TOOL, FINISH, ToolSpec
from lifemate.config import PromptConfig
from lifemate.data import Sample, Split


def trajectory_messages(
    builder: PromptBuilder,
    sample: Sample,
    transcripts: dict[str, str],
    examples: str = "",
    specs: list[ToolSpec] | None = None,
    apps: dict | None = None,
) -> list[dict[str, str]]:
    cfg = builder.cfg
    messages, _ = builder.build(sample, transcripts, examples, specs=specs, apps=apps)
    calls = sample.gold_calls or []
    results = (sample.label or {}).get("tool_results") or []
    for i, c in enumerate(calls):
        messages.append({"role": "assistant", "content": format_action(c["name"], c.get("arguments"))})
        if c["name"] == ASK_TOOL:
            return messages
        result = results[i] if i < len(results) else {"ok": True}
        messages.append({"role": "user", "content": format_observation(c["name"], result, cfg.result_max_items, cfg.result_max_chars)})
    messages.append({"role": "assistant", "content": format_action(FINISH)})
    return messages


def iter_examples(
    split: Split,
    transcripts: dict[str, str],
    cfg: PromptConfig | None = None,
    retriever: Retriever | None = None,
    ids: set[str] | None = None,
    exclude_from_examples: dict[str, set[str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """``exclude_from_examples[id]`` keeps validation samples out of the few-shot pool of a training example."""
    builder = PromptBuilder(split, cfg)
    for s in split:
        if ids is not None and s.id not in ids:
            continue
        examples = ""
        if retriever is not None and builder.cfg.fewshot_k > 0:
            exclude = {s.id} | (exclude_from_examples or {}).get(s.id, set())
            request = transcripts.get(s.audio, s.audio_text) or ""
            examples = render_examples(retriever.query(request, builder.cfg.fewshot_k, exclude=exclude))
        yield {"id": s.id, "messages": trajectory_messages(builder, s, transcripts, examples)}


def write_jsonl(rows: Iterator[dict[str, Any]] | list[dict[str, Any]], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_augmented(
    split: Split,
    transcripts: dict[str, str],
    copies: int,
    cfg: PromptConfig | None = None,
    ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """``copies`` tool-renamed variants per sample (no few-shot block: examples would show the original names)."""
    from lifemate.train.augment import rename_sample

    builder = PromptBuilder(split, cfg)
    for s in split:
        if ids is not None and s.id not in ids:
            continue
        for k in range(1, copies + 1):
            new, specs, apps = rename_sample(s, builder.specs_for(s), split.apps, seed=k)
            yield {"id": new.id, "messages": trajectory_messages(builder, new, transcripts, "", specs=specs, apps=apps)}
