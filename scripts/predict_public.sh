#!/usr/bin/env bash
# End-to-end submission for a test split: ASR (cached) -> vLLM server -> agent -> schema validation.
#
#   bash scripts/predict_public.sh                                  # base model Qwen3.5-4B
#   LORA=checkpoints/sft-v1 bash scripts/predict_public.sh          # base + LoRA adapter
#   MODEL=models/sft-v1 bash scripts/predict_public.sh              # merged checkpoint
#
# Settings (environment variables): SPLIT, MODEL, LORA, ASR_MODEL, PORT, MEM, FEWSHOT, OUT_DIR, TRANSCRIPTS.
# Output: outputs/<split>/submission_<tag>.jsonl, ready to upload.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT=${SPLIT:-public_test}
MODEL=${MODEL:-models/base/Qwen3.5-4B}
LORA=${LORA:-}
ASR_MODEL=${ASR_MODEL:-models/base/Qwen3-ASR-0.6B}
PORT=${PORT:-8011}
MEM=${MEM:-0.85}
FEWSHOT=${FEWSHOT:-3}
OUT_DIR=${OUT_DIR:-outputs/$SPLIT}
TAG=$(basename "${LORA:-$MODEL}")
mkdir -p "$OUT_DIR"

if [[ ! -e "$ASR_MODEL" && "$ASR_MODEL" == models/* ]]; then
  echo "$ASR_MODEL not found locally, using Qwen/Qwen3-ASR-0.6B from the Hub"
  ASR_MODEL=Qwen/Qwen3-ASR-0.6B
fi
ASR_TAG=$(echo "$ASR_MODEL" | sed 's#^models/base/##; s#^checkpoints/##; s#[/:]#_#g')
TRANSCRIPTS=${TRANSCRIPTS:-$OUT_DIR/asr_${ASR_TAG}.jsonl}
SUB="$OUT_DIR/submission_${TAG}.jsonl"

echo "== 1/3 ASR ($ASR_MODEL)"
if [[ -s "$TRANSCRIPTS" ]]; then
  echo "using cached transcripts: $TRANSCRIPTS"
else
  lifemate transcribe --split "$SPLIT" --asr qwen --asr-model "$ASR_MODEL" --out "$TRANSCRIPTS"
fi

echo "== 2/3 vLLM ($MODEL${LORA:+ + LoRA $LORA})"
NAME=lifemate
if [[ -n "$LORA" ]]; then
  source scripts/serve.sh "$MODEL" base "$PORT" "$MEM" --enable-lora --max-lora-rank 64 --lora-modules "$NAME=$LORA"
else
  source scripts/serve.sh "$MODEL" "$NAME" "$PORT" "$MEM"
fi
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "== 3/3 agent -> $SUB"
lifemate infer --split "$SPLIT" --backend openai --model "$NAME" --base-url "$BASE_URL" \
  --transcripts "$TRANSCRIPTS" --fewshot-k "$FEWSHOT" \
  --out "$SUB" --trace "${SUB%.jsonl}.trace.jsonl"
lifemate validate --split "$SPLIT" --submission "$SUB"

echo
echo "submission ready: $SUB ($(wc -l < "$SUB") lines)"
