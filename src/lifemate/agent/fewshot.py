"""Retrieval of similar training trajectories as in-context examples (TF-IDF on folded words and bigrams)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable

from lifemate.agent.render import compact_result, format_action
from lifemate.data import Sample
from lifemate.text import fold


def _features(text: str) -> Counter:
    words = fold(text).replace(",", " ").replace(".", " ").split()
    feats = Counter(words)
    feats.update(f"{a}_{b}" for a, b in zip(words, words[1:]))
    return feats


class Retriever:
    def __init__(self, samples: list[Sample], text_of: Callable[[Sample], str] | None = None):
        self.samples = [s for s in samples if s.label is not None]
        self.text_of = text_of or (lambda s: s.audio_text or "")
        docs = [_features(self.text_of(s)) for s in self.samples]
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, d in enumerate(docs):
            vec = {t: c * self.idf[t] for t, c in d.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for t, v in vec.items():
                self.index[t].append((i, v / norm))

    def query(self, text: str, k: int, exclude: set[str] | None = None) -> list[Sample]:
        exclude = exclude or set()
        q = {t: c * self.idf.get(t, 0.0) for t, c in _features(text).items()}
        norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for t, w in q.items():
            for i, v in self.index.get(t, ()):
                scores[i] += w / norm * v
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.samples[kv[0]].id))
        out = []
        for i, _ in ranked:
            s = self.samples[i]
            if s.id in exclude:
                continue
            out.append(s)
            if len(out) == k:
                break
        return out


def render_examples(examples: list[Sample], result_chars: int = 220) -> str:
    blocks = []
    for n, s in enumerate(examples, 1):
        lines = [f"Ví dụ {n}: \"{s.audio_text}\""]
        calls = s.gold_calls or []
        results = (s.label or {}).get("tool_results") or []
        for i, c in enumerate(calls):
            line = f"  → {format_action(c['name'], c.get('arguments'))}"
            if i < len(results) and c["name"] != "sys.ask_missing_info":
                line += f"  ⇒ {compact_result(results[i], max_items=4, max_chars=result_chars)}"
            lines.append(line)
        lines.append(f"  → {format_action('finish')}" if not calls or calls[-1]["name"] != "sys.ask_missing_info" else "")
        blocks.append("\n".join(x for x in lines if x))
    return "\n".join(blocks)
