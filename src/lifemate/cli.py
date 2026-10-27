"""Command line interface: ``lifemate <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

log = logging.getLogger("lifemate")


# -- helpers -------------------------------------------------------------------------------------
def _split(args):
    from lifemate.data import Split

    return Split.resolve(args.data, args.split)


def _transcripts(args, split):
    from lifemate.asr import TranscriptStore, oracle_transcripts

    if args.transcripts:
        return TranscriptStore.load(args.transcripts)
    if args.asr == "oracle":
        if not split.has_labels:
            sys.exit("--asr oracle needs a labelled split; pass --transcripts for test splits")
        return oracle_transcripts(split)
    from lifemate.asr.base import transcribe_split
    from lifemate.asr.qwen import QwenASR

    asr = QwenASR(args.asr_model, backend=args.asr_backend)
    store = transcribe_split(split, asr)
    asr.close()
    return store


def _read_ids(path):
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def _ids(args, split):
    ids = [s.id for s in split]
    if args.ids:
        wanted = set(args.ids.split(","))
        ids = [i for i in ids if i in wanted]
    if getattr(args, "ids_file", None):
        wanted = _read_ids(args.ids_file)
        ids = [i for i in ids if i in wanted]
    if args.limit:
        ids = ids[: args.limit]
    return ids


DEFAULT_DATA = os.environ.get("LIFEMATE_DATA", "data")


def _add_data(p, split_default="train"):
    p.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="split folder, or a folder containing <split>/ (default: $LIFEMATE_DATA or ./data)",
    )
    p.add_argument("--split", default=split_default)


# -- commands ------------------------------------------------------------------------------------
def cmd_stats(args) -> None:
    split = _split(args)
    kinds = {n: t.get("kind") for n, t in split.tools.items()}
    print(f"{split}: {len(split)} samples, {len(split.apps)} apps, {len(split.tools)} tools")
    print(f"with history: {sum(bool(s.history) for s in split)}")
    if split.has_labels:
        n_calls = Counter(len(s.gold_calls) for s in split)
        print("gold calls per sample:", dict(sorted(n_calls.items())))
        print("out of scope ([]):", n_calls[0])
        print("with sys.ask_missing_info:", sum(any(c["name"] == "sys.ask_missing_info" for c in s.gold_calls) for s in split))
        writes = Counter(sum(kinds.get(c["name"]) == "write" for c in s.gold_calls) for s in split)
        print("writes per sample:", dict(sorted(writes.items())))


def cmd_transcribe(args) -> None:
    split = _split(args)
    store = _transcripts(args, split)
    store.save(args.out)
    print(f"{len(store)} transcripts -> {args.out}")


def cmd_infer(args) -> None:
    from lifemate.agent import Agent
    from lifemate.agent.fewshot import Retriever
    from lifemate.config import load_agent_config
    from lifemate.data import load_samples
    from lifemate.pipeline import run

    split = _split(args)
    cfg = load_agent_config(args.config)
    if args.fewshot_k is not None:
        cfg.prompt.fewshot_k = args.fewshot_k
    if args.no_constrained:
        cfg.constrained = False
    transcripts = _transcripts(args, split)

    retriever = None
    if cfg.prompt.fewshot_k > 0:
        pool = Path(args.examples) if args.examples else (split.dir.parent / "train" / "train.jsonl")
        if pool.is_dir():
            pool = pool / "train.jsonl"
        if pool.exists():
            excluded = _read_ids(args.exclude_ids) if args.exclude_ids else set()
            retriever = Retriever([x for x in load_samples(pool) if x.id not in excluded])
        else:
            log.warning("no few-shot pool at %s; running without examples", pool)

    if args.backend == "oracle":
        from lifemate.agent.llm import OracleBackend

        backend = OracleBackend()
    elif args.backend == "openai":
        from lifemate.agent.llm import OpenAICompatBackend

        backend = OpenAICompatBackend(args.base_url, args.model, api_key=args.api_key)
        backend.check()
    else:
        from lifemate.agent.llm import VLLMBackend

        backend = VLLMBackend(args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, lora_path=args.lora)

    agent = Agent(split, backend, cfg, retriever)
    wanted = set(_ids(args, split))
    stats = run(agent, (s for s in split if s.id in wanted), transcripts, args.out, args.trace, args.resume)
    print(json.dumps(stats, indent=2))


def cmd_score(args) -> None:
    from lifemate.eval.report import format_report, taxonomy
    from lifemate.eval.scorer import Scorer, dump_scores, load_submission, summarize

    split = _split(args)
    scorer = Scorer(split, lenient_args=args.lenient)
    preds = load_submission(args.submission)
    ids = list(preds) if args.only_submitted else None
    if args.ids_file:
        wanted = _read_ids(args.ids_file)
        ids = [s.id for s in split if s.id in wanted]
    scores = scorer.score_predictions(preds, ids)
    kinds = {n: t.get("kind", "write") for n, t in split.tools.items()}
    print(format_report(summarize(scores), taxonomy(scores, kinds), len(scores)))
    if args.report:
        dump_scores(scores, args.report)
        print(f"per-sample details -> {args.report}")


def cmd_validate(args) -> None:
    split = _split(args)
    seen, problems = set(), 0
    with open(args.submission, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            row = json.loads(line)
            sid = row.get("id")
            if sid in seen:
                print(f"line {n}: duplicate id {sid}")
                problems += 1
            seen.add(sid)
            if not isinstance(row.get("state_hash"), str):
                print(f"line {n}: missing state_hash")
                problems += 1
            allowed = set(split.get(sid).installed_tools) | {"sys.ask_missing_info"}
            for r in split.zaic.schema_validate(row.get("tool_calls") or [], split.catalog, allowed):
                if not r["ok"]:
                    print(f"{sid}: {r['name']}: {r['error']}")
                    problems += 1
    missing = {s.id for s in split} - seen
    if missing:
        print(f"{len(missing)} sample ids missing, e.g. {sorted(missing)[:5]}")
    print("OK" if not problems and not missing else f"{problems} problems, {len(missing)} missing")


def cmd_build_sft(args) -> None:
    from lifemate.agent.fewshot import Retriever
    from lifemate.config import PromptConfig
    from lifemate.eval.splits import group_kfold, leave_apps_out, pick_held_out_apps
    from lifemate.train.sft_data import iter_augmented, iter_examples, write_jsonl

    split = _split(args)
    transcripts = _transcripts(args, split)
    cfg = PromptConfig(fewshot_k=args.fewshot_k)
    if args.holdout == "apps":
        held = pick_held_out_apps(split.samples, args.n_apps, args.seed)
        train_ids, val_ids = map(set, leave_apps_out(split.samples, held))
        print("held-out apps:", ", ".join(sorted(held)))
    elif args.holdout == "fold":
        folds = group_kfold(split.samples, args.k, args.seed)
        val_ids = {i for i, f in folds.items() if f == args.fold}
        train_ids = set(folds) - val_ids
    else:
        train_ids, val_ids = {s.id for s in split}, set()
    retriever = None
    if args.fewshot_k > 0:  # the few-shot pool of training rows never contains validation samples
        retriever = Retriever([s for s in split if s.id in train_ids])
    out = Path(args.out_dir)
    rows = list(iter_examples(split, transcripts, cfg, retriever, ids=train_ids))
    if args.augment:
        rows += list(iter_augmented(split, transcripts, args.augment, cfg, ids=train_ids))
    print(f"train: {write_jsonl(rows, out / 'train.jsonl')} rows")
    if val_ids:
        val_rows = iter_examples(split, transcripts, cfg, retriever, ids=val_ids)
        print(f"val:   {write_jsonl(val_rows, out / 'val.jsonl')} rows")
        (out / "val_ids.txt").write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")


def cmd_asr_eval(args) -> None:
    from lifemate.asr import TranscriptStore, oracle_transcripts
    from lifemate.asr.metrics import number_error_rate, wer, wer_no_tone

    split = _split(args)
    gold = oracle_transcripts(split)
    hyp = TranscriptStore.load(args.transcripts)
    keys = sorted(set(gold) & set(hyp))
    if args.ids_file:  # keep only utterances of the listed samples (audio/<id>/...)
        wanted = _read_ids(args.ids_file)
        keys = [k for k in keys if k.split("/")[1] in wanted]
    refs, hyps = [gold[k] for k in keys], [hyp[k] for k in keys]
    print(f"utterances        {len(keys)}")
    print(f"WER               {100 * wer(refs, hyps):.2f}%")
    print(f"WER without tones {100 * wer_no_tone(refs, hyps):.2f}%")
    print(f"number errors     {100 * number_error_rate(refs, hyps):.2f}% of utterances with digits")


def cmd_asr_manifest(args) -> None:
    from lifemate.train.asr_data import build_manifests

    n_train, n_val = build_manifests(_split(args), args.out_dir, val_ids_path=args.val_ids)
    print(f"{n_train} train / {n_val} val utterances -> {args.out_dir}")


# -- parser --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lifemate", description="LifeMate voice-to-action agent (Zalo AI Challenge 2026)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("stats", help="dataset statistics")
    _add_data(s)
    s.set_defaults(func=cmd_stats)

    def add_asr(q):
        q.add_argument("--asr", choices=["oracle", "qwen"], default="qwen")
        q.add_argument("--asr-model", default="Qwen/Qwen3-ASR-0.6B")
        q.add_argument("--asr-backend", choices=["transformers", "vllm"], default="transformers")
        q.add_argument("--transcripts", default=None, help="pre-computed transcripts (JSON Lines); skips ASR")

    s = sub.add_parser("transcribe", help="run ASR on every user turn of a split")
    _add_data(s, "public_test")
    add_asr(s)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("infer", help="run the agent and write a submission")
    _add_data(s, "public_test")
    add_asr(s)
    s.add_argument("--backend", choices=["vllm", "openai", "oracle"], default="vllm")
    s.add_argument("--model", default="Qwen/Qwen3.5-4B")
    s.add_argument("--lora", default=None, help="LoRA adapter (vllm backend)")
    s.add_argument("--base-url", default="http://localhost:8000/v1")
    s.add_argument("--api-key", default="EMPTY")
    s.add_argument("--max-model-len", type=int, default=16384)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    s.add_argument("--config", default=None, help="agent config (YAML or JSON)")
    s.add_argument("--fewshot-k", type=int, default=None)
    s.add_argument("--no-constrained", action="store_true", help="disable JSON-schema constrained decoding")
    s.add_argument("--examples", default=None, help="few-shot pool: train.jsonl or its folder")
    s.add_argument("--out", required=True)
    s.add_argument("--trace", default=None, help="per-sample step trace (JSON Lines)")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--ids", default=None, help="comma-separated sample ids")
    s.add_argument("--ids-file", default=None, help="run only the ids listed in this file (one per line)")
    s.add_argument("--exclude-ids", default=None, help="ids removed from the few-shot pool (e.g. a validation split)")
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("score", help="score a submission on a labelled split")
    _add_data(s)
    s.add_argument("--submission", required=True)
    s.add_argument("--report", default=None, help="per-sample scores (JSON Lines)")
    s.add_argument("--lenient", action="store_true", help="case/whitespace-insensitive non-fuzzy arguments")
    s.add_argument("--only-submitted", action="store_true", help="score only the ids present in the submission")
    s.add_argument("--ids-file", default=None, help="score only the ids listed in this file")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("validate", help="schema-check a submission with the split's zaic_tools")
    _add_data(s, "public_test")
    s.add_argument("--submission", required=True)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("build-sft", help="render gold trajectories as SFT chat data")
    _add_data(s)
    add_asr(s)
    s.add_argument("--out-dir", default="outputs/sft")
    s.add_argument("--fewshot-k", type=int, default=3)
    s.add_argument("--augment", type=int, default=0, help="tool-renamed copies per sample")
    s.add_argument("--holdout", choices=["none", "fold", "apps"], default="apps")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--n-apps", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_build_sft)

    s = sub.add_parser("asr-eval", help="WER and number errors of transcripts against gold (train)")
    _add_data(s)
    s.add_argument("--transcripts", required=True)
    s.add_argument("--ids-file", default=None, help="evaluate only the utterances of these sample ids")
    s.set_defaults(func=cmd_asr_eval)

    s = sub.add_parser("asr-manifest", help="ASR fine-tuning manifests from train")
    _add_data(s)
    s.add_argument("--out-dir", default="outputs/asr")
    s.add_argument("--val-ids", default=None, help="sample ids for the validation manifest (default: 5%% hashed)")
    s.set_defaults(func=cmd_asr_manifest)
    return p


def main(argv: list[str] | None = None) -> int:
    from lifemate.agent.llm import BackendError
    from lifemate.pipeline import TooManyErrors

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        args.func(args)
    except (BackendError, TooManyErrors) as e:
        log.error("%s", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
