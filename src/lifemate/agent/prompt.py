"""Builds the chat messages for one sample."""

from __future__ import annotations

from collections.abc import Sequence

from lifemate.agent.policy import SYSTEM_PROMPT
from lifemate.agent.render import compact_result, dumps, format_action
from lifemate.catalog import ASK_TOOL, ToolSpec, load_specs, render_toolbox
from lifemate.config import PromptConfig
from lifemate.data import Sample, Split
from lifemate.timeutil import WEEKDAYS_VI, day_table, parse_now

UNHEARD = "(không nghe rõ)"


class PromptBuilder:
    def __init__(self, split: Split, cfg: PromptConfig | None = None):
        self.split = split
        self.cfg = cfg or PromptConfig()
        self.specs = load_specs(split.tools)

    def specs_for(self, sample: Sample) -> list[ToolSpec]:
        """Installed tools in catalog order of the sample, plus sys.ask_missing_info."""
        names = [t for t in sample.installed_tools if t in self.specs]
        if ASK_TOOL in self.specs and ASK_TOOL not in names:
            names.append(ASK_TOOL)
        return [self.specs[n] for n in names]

    def system_message(self, specs: Sequence[ToolSpec], apps: dict | None = None) -> str:
        return f"{SYSTEM_PROMPT}\n\n# Công cụ\n{render_toolbox(list(specs), self.split.apps if apps is None else apps)}"

    def context_block(self, sample: Sample) -> str:
        ctx = sample.context
        now = parse_now(ctx["current_time"])
        lines = [
            f"Thời điểm hiện tại: {ctx['current_time']} ({WEEKDAYS_VI[now.weekday()]}), múi giờ {ctx.get('timezone', '')}",
            day_table(now, self.cfg.days_ahead),
        ]
        if loc := ctx.get("location"):
            lines.append(f"Vị trí hiện tại: {dumps(loc)}")
        if profile := ctx.get("user_profile"):
            lines.append(f"Hồ sơ người dùng: {dumps(profile)}")
        return "\n".join(lines)

    def history_block(self, sample: Sample, transcripts: dict[str, str]) -> str:
        lines = []
        for turn in sample.history:
            if turn.get("role") == "user":
                text = transcripts.get(turn.get("audio", ""), turn.get("audio_text")) or UNHEARD
                lines.append(f'Người dùng: "{text}"')
                continue
            calls = turn.get("tool_calls") or []
            results = turn.get("tool_results") or []
            for i, c in enumerate(calls):
                line = f"Trợ lý gọi: {format_action(c['name'], c.get('arguments'))}"
                if i < len(results):
                    line += f" → {compact_result(results[i], self.cfg.result_max_items, self.cfg.history_result_max_chars)}"
                lines.append(line)
            if turn.get("text"):
                lines.append(f"Trợ lý: {turn['text']}")
        return "\n".join(lines)

    def user_message(self, sample: Sample, transcripts: dict[str, str], examples: str = "") -> str:
        request = transcripts.get(sample.audio, sample.audio_text) or UNHEARD
        parts = ["# Bối cảnh", self.context_block(sample)]
        if examples:
            parts += ["", "# Ví dụ tham khảo (từ các tình huống khác)", examples]
        history = self.history_block(sample, transcripts)
        parts += ["", "# Lịch sử hội thoại", history or "(chưa có)"]
        parts += ["", "# Yêu cầu cần xử lý", f'Người dùng: "{request}"']
        return "\n".join(parts)

    def build(
        self,
        sample: Sample,
        transcripts: dict[str, str],
        examples: str = "",
        specs: list[ToolSpec] | None = None,
        apps: dict | None = None,
    ) -> tuple[list[dict], list[ToolSpec]]:
        """System + user messages. ``specs``/``apps`` override the catalog (used by tool-renaming augmentation)."""
        specs = self.specs_for(sample) if specs is None else specs
        messages = [
            {"role": "system", "content": self.system_message(specs, apps)},
            {"role": "user", "content": self.user_message(sample, transcripts, examples)},
        ]
        return messages, specs
