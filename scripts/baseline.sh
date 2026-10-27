#!/usr/bin/env bash
# Zero-shot / few-shot baseline on train (gold transcripts) with a local vLLM server.
#   bash scripts/baseline.sh ../train+public_test Qwen/Qwen3.5-4B
set -euo pipefail
DATA=${1:-../train+public_test}
MODEL=${2:-Qwen/Qwen3.5-4B}
PORT=${PORT:-8000}
mkdir -p outputs

vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 --enable-prefix-caching \
  --gpu-memory-utilization 0.85 --seed 0 > outputs/vllm.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do sleep 5; done

for K in 0 3; do
  lifemate infer --data "$DATA" --split train --backend openai --model "$MODEL" \
    --base-url "http://localhost:$PORT/v1" --asr oracle --fewshot-k "$K" \
    --out "outputs/baseline_k$K.jsonl" --trace "outputs/baseline_k$K.trace.jsonl"
  lifemate score --data "$DATA" --split train --submission "outputs/baseline_k$K.jsonl" \
    --report "outputs/baseline_k$K.scores.jsonl"
done
