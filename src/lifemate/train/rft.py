"""Rejection-sampling fine-tuning data: sample several trajectories per training task, keep the perfect ones.

A cheap stand-in for RL on a single GPU: it needs only inference (a ``vllm serve`` of the current model) and the
offline scorer. Kept trajectories are clean (no guard rejection, no parse error) and are written in the SFT format.

    vllm serve <model-or-merged-checkpoint> --max-model-len 16384 --enable-prefix-caching
    python -m lifemate.train.rft --data ../train+public_test --base-url http://localhost:8000/v1 --model <name> --samples 4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lifemate.agent import Agent
from lifemate.agent.fewshot import Retriever
from lifemate.agent.llm import OpenAICompatBackend
from lifemate.asr import TranscriptStore, oracle_transcripts
from lifemate.config import AgentConfig
from lifemate.data import Split
from lifemate.eval.scorer import Scorer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=4, help="trajectories sampled per task")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--transcripts", default=None, help="ASR transcripts (default: gold transcripts)")
    ap.add_argument("--fewshot-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="outputs/rft/train.jsonl")
    args = ap.parse_args(argv)

    split = Split.resolve(args.data, args.split)
    scorer = Scorer(split)
    transcripts = TranscriptStore.load(args.transcripts) if args.transcripts else oracle_transcripts(split)
    cfg = AgentConfig(temperature=args.temperature)
    cfg.prompt.fewshot_k = args.fewshot_k
    retriever = Retriever(split.samples) if args.fewshot_k > 0 else None

    def solve(job: tuple[str, int]) -> dict | None:
        sample_id, k = job
        backend = OpenAICompatBackend(args.base_url, args.model, seed=k)
        result = Agent(split, backend, cfg, retriever).run(split.get(sample_id), transcripts)
        clean = all(step.executed or (step.action or {}).get("name") == "finish" for step in result.steps)
        if not clean or scorer.score_sample(sample_id, result.tool_calls).score < 1.0:
            return None
        return {"id": f"{sample_id}#s{k}", "messages": result.messages}

    jobs = [(s.id, k) for s in split for k in range(args.samples)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept, seen = 0, set()
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "w", encoding="utf-8") as f:
        for row in pool.map(solve, jobs):
            if row is None:
                continue
            key = json.dumps(row["messages"][2:], ensure_ascii=False)
            if key in seen:  # identical trajectories add nothing
                continue
            seen.add(key)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept} distinct perfect trajectories from {len(jobs)} rollouts -> {args.out}")


if __name__ == "__main__":
    main()
