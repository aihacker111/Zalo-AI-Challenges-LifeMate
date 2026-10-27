#!/usr/bin/env bash
# Zero-shot / few-shot baseline on train (gold transcripts) with a local vLLM server.
#   bash scripts/baseline.sh data Qwen/Qwen3.5-4B
set -euo pipefail
DATA=${1:-data}
MODEL=${2:-Qwen/Qwen3.5-4B}
PORT=${PORT:-8011}
MEM=${MEM:-0.85}
mkdir -p outputs

source "$(dirname "$0")/serve.sh" "$MODEL" base "$PORT" "$MEM"
trap 'kill $VLLM_PID' EXIT

for K in 0 3; do
  lifemate infer --data "$DATA" --split train --backend openai --model base \
    --base-url "$BASE_URL" --asr oracle --fewshot-k "$K" \
    --out "outputs/baseline_k$K.jsonl" --trace "outputs/baseline_k$K.trace.jsonl"
  lifemate score --data "$DATA" --split train --submission "outputs/baseline_k$K.jsonl" \
    --report "outputs/baseline_k$K.scores.jsonl"
done
