#!/usr/bin/env bash
# Compare ASR engines on the validation utterances (878 user turns of the 274 val samples).
#   bash scripts/asr_benchmark.sh                                  # gipformer, chunkformer, qwen (base)
#   ENGINES="gipformer qwen:checkpoints/asr-ft/checkpoint-390" bash scripts/asr_benchmark.sh
#   ENGINES="gipformer gipformer:models/gipformer-ft-e10" bash scripts/asr_benchmark.sh   # fine-tuned gipformer
# An entry is "engine" or "engine:model". Transcripts go to outputs/asr_bench/<name>.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
VAL=${VAL:-outputs/asr_ft/val_ids.txt}
[[ -s "$VAL" ]] || VAL=outputs/sft/val_ids.txt
ENGINES=${ENGINES:-"gipformer chunkformer qwen:models/base/Qwen3-ASR-0.6B"}
OUT=outputs/asr_bench
mkdir -p "$OUT"

for entry in $ENGINES; do
  engine=${entry%%:*}
  model=""; [[ "$entry" == *:* ]] && model=${entry#*:}
  name=$(echo "${engine}_${model:-default}" | sed 's#[/:]#_#g')
  out="$OUT/$name.jsonl"
  echo "== $engine ${model:-(default model)}"
  # Re-transcribe when the model directory was re-exported after the cached transcripts were written.
  if [[ -s "$out" && -n "$model" && -f "$model/encoder.onnx" && "$model/encoder.onnx" -nt "$out" ]]; then
    echo "   $model was re-exported: transcribing again"; rm -f "$out"
  fi
  if [[ ! -s "$out" ]]; then
    start=$(date +%s)
    lifemate transcribe --split train --asr "$engine" ${model:+--asr-model "$model"} --ids-file "$VAL" --out "$out"
    echo "   time: $(( $(date +%s) - start ))s"
  fi
  lifemate asr-eval --split train --transcripts "$out" --ids-file "$VAL" | sed 's/^/   /'
done
