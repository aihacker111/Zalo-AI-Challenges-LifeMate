"""The agent: one session per sample, one model step per tool call, the guard in between."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lifemate.agent.fewshot import Retriever, render_examples
from lifemate.agent.guard import Guard
from lifemate.agent.parser import Action, ParseError, parse_action
from lifemate.agent.prompt import PromptBuilder
from lifemate.agent.render import format_action, format_observation, format_rejection
from lifemate.catalog import ASK_TOOL, action_schema
from lifemate.config import AgentConfig
from lifemate.data import Sample, Split
from lifemate.timeutil import parse_now, utc_offset


@dataclass
class Step:
    output: str
    action: dict[str, Any] | None
    executed: bool
    observation: str


@dataclass
class AgentResult:
    id: str
    tool_calls: list[dict[str, Any]]
    state_hash: str
    steps: list[Step] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None

    def submission_line(self) -> dict[str, Any]:
        return {"id": self.id, "tool_calls": self.tool_calls, "state_hash": self.state_hash}


class Agent:
    def __init__(
        self,
        split: Split,
        backend: Any,
        config: AgentConfig | None = None,
        retriever: Retriever | None = None,
        stop_after_ask: bool = True,
    ):
        self.split = split
        self.backend = backend
        self.cfg = config or AgentConfig()
        self.prompts = PromptBuilder(split, self.cfg.prompt)
        self.retriever = retriever
        self.stop_after_ask = stop_after_ask

    def examples_for(self, sample: Sample, request: str) -> str:
        if self.retriever is None or self.cfg.prompt.fewshot_k <= 0:
            return ""
        found = self.retriever.query(request, self.cfg.prompt.fewshot_k, exclude={sample.id})
        return render_examples(found)

    def run(self, sample: Sample, transcripts: dict[str, str] | None = None) -> AgentResult:
        t0 = time.monotonic()
        transcripts = transcripts or {}
        session = self.split.open_session(sample.id)  # exactly one session per sample
        if hasattr(self.backend, "begin"):
            self.backend.begin(sample)
        request = transcripts.get(sample.audio, sample.audio_text) or ""
        messages, specs = self.prompts.build(sample, transcripts, self.examples_for(sample, request))
        schema = action_schema(specs) if self.cfg.constrained else None
        offset = utc_offset(parse_now(sample.context["current_time"]))
        guard = Guard.for_sample(session, specs, sample.history, offset, self.cfg.guard)
        steps: list[Step] = []
        error = None
        pc = self.cfg.prompt
        try:
            for _ in range(self.cfg.max_steps):
                if time.monotonic() - t0 > self.cfg.time_budget_s:
                    error = "time budget exhausted"
                    break
                output = self.backend.generate(messages, schema, self.cfg.temperature, self.cfg.max_new_tokens)
                try:
                    action = parse_action(output)
                except ParseError as e:
                    messages += [{"role": "assistant", "content": output}, {"role": "user", "content": format_rejection(str(e))}]
                    steps.append(Step(output, None, False, str(e)))
                    continue
                messages.append({"role": "assistant", "content": format_action(action.name, action.arguments)})
                if action.is_finish:
                    steps.append(Step(output, {"name": action.name}, False, ""))
                    break
                obs, executed, final = self._act(session, guard, action, pc.result_max_items, pc.result_max_chars)
                steps.append(Step(output, {"name": action.name, "arguments": action.arguments}, executed, obs))
                if final:  # sys.ask_missing_info ends the turn: the user has to answer first
                    break
                messages.append({"role": "user", "content": obs})
        except Exception as e:  # never lose the calls already made
            error = f"{type(e).__name__}: {e}"
        return AgentResult(
            id=sample.id,
            tool_calls=session.get_tool_calls(),
            state_hash=session.get_state_hash(),
            steps=steps,
            messages=messages,
            elapsed_s=time.monotonic() - t0,
            error=error,
        )

    def _act(self, session: Any, guard: Guard, action: Action, max_items: int, max_chars: int) -> tuple[str, bool, bool]:
        decision = guard.review(action)
        if not decision.execute:
            text = decision.message
            if decision.cached is not None:
                text += ": " + guard.cached_text(decision, max_chars)
            return format_rejection(text), False, False
        result = session.call(decision.action.name, decision.action.arguments)
        guard.record(decision.action, result)
        final = self.stop_after_ask and decision.action.name == ASK_TOOL
        return format_observation(decision.action.name, result, max_items, max_chars), True, final
