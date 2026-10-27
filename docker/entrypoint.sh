#!/usr/bin/env bash
# Organisers mount the test split at /data and collect /app/result/submission.jsonl.
set -euo pipefail
DATA=${DATA:-/data}
OUT=${OUT:-/app/result}
mkdir -p "$OUT"

# 1) ASR on every user turn, then the ASR model is released before the LLM loads.
lifemate transcribe --data "$DATA" --asr "${ASR:-gipformer}" --asr-model /models/asr --out "$OUT/transcripts.jsonl"

# 2) Agent with the merged LLM, greedy decoding, one ToolEnv session per sample.
lifemate infer --data "$DATA" --backend vllm --model /models/llm \
  --transcripts "$OUT/transcripts.jsonl" --examples /app/assets/train/train.jsonl \
  --config /app/configs/agent.yaml --out "$OUT/submission.jsonl" --trace "$OUT/trace.jsonl"

lifemate validate --data "$DATA" --submission "$OUT/submission.jsonl"
