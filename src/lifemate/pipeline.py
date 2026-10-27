"""Run the agent over a split and write a submission (resumable, flushed line by line)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from lifemate.agent import Agent
from lifemate.data import Sample

log = logging.getLogger("lifemate")


class TooManyErrors(RuntimeError):
    """Raised when consecutive samples fail, so a broken backend never yields a full but empty submission."""


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def run(
    agent: Agent,
    samples: Iterable[Sample],
    transcripts: dict[str, str],
    out: str | Path,
    trace: str | Path | None = None,
    resume: bool = False,
    max_consecutive_errors: int = 5,
) -> dict[str, float]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out) if resume else set()
    mode = "a" if resume else "w"
    n = errors = streak = 0
    slowest = 0.0
    t0 = time.monotonic()
    ftrace = open(trace, mode, encoding="utf-8") if trace else None
    try:
        with open(out, mode, encoding="utf-8") as fout:
            for sample in samples:
                if sample.id in skip:
                    continue
                result = agent.run(sample, transcripts)
                fout.write(json.dumps(result.submission_line(), ensure_ascii=False) + "\n")
                fout.flush()
                if ftrace:
                    ftrace.write(
                        json.dumps(
                            {
                                "id": result.id,
                                "elapsed_s": round(result.elapsed_s, 3),
                                "error": result.error,
                                "steps": [s.__dict__ for s in result.steps],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                n += 1
                errors += result.error is not None
                streak = streak + 1 if result.error else 0
                if max_consecutive_errors and streak >= max_consecutive_errors:
                    raise TooManyErrors(f"{streak} consecutive samples failed, last: {result.error}")
                slowest = max(slowest, result.elapsed_s)
                if result.error:
                    log.warning("%s: %s", sample.id, result.error)
                if n % 50 == 0:
                    log.info("%d samples, %.2fs/sample", n, (time.monotonic() - t0) / n)
    finally:
        if ftrace:
            ftrace.close()
    total = time.monotonic() - t0
    return {"samples": n, "errors": errors, "seconds_per_sample": total / max(n, 1), "slowest_s": slowest}
