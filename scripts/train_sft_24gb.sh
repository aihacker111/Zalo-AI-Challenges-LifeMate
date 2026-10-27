#!/usr/bin/env bash
# From raw data to a trained LoRA adapter on one 24 GB GPU.
#
#   bash scripts/train_sft_24gb.sh                     # HOLDOUT=apps: keeps a validation set (checkpoints/sft-v1)
#   HOLDOUT=none bash scripts/train_sft_24gb.sh        # all training data (checkpoints/sft-full)
#
# Steps (each is skipped when its output already exists):
#   1. ASR transcripts of every user turn in train (base Qwen3-ASR, never a model fine-tuned on train audio)
#   2. SFT data from gold transcripts and from ASR transcripts, mixed and shuffled
#   3. LoRA SFT
# Settings: HOLDOUT, NAME, EPOCHS, MIX_ASR, ASR, ASR_MODEL, AUGMENT_GOLD, AUGMENT_ASR, FEWSHOT, MAX_LENGTH, BASE_MODEL.
# MIX_ASR=1 adds a copy of the data whose inputs are ASR transcripts (default engine: gipformer, ~16% WER on val and
# never trained on this audio, so its errors are realistic). Do not use an ASR fine-tuned on the train audio here:
# its train transcripts are cleaner than what it produces on unseen audio. MIX_ASR=0 (default) uses gold only.
set -euo pipefail
cd "$(dirname "$0")/.."

HOLDOUT=${HOLDOUT:-apps}
if [[ "$HOLDOUT" == none ]]; then DEFAULT_NAME=sft-full; else DEFAULT_NAME=sft-v1; fi
NAME=${NAME:-$DEFAULT_NAME}
EPOCHS=${EPOCHS:-1}
MIX_ASR=${MIX_ASR:-0}
AUGMENT_GOLD=${AUGMENT_GOLD:-1}
AUGMENT_ASR=${AUGMENT_ASR:-0}
FEWSHOT=${FEWSHOT:-3}
MAX_LENGTH=${MAX_LENGTH:-12288}
BASE_MODEL=${BASE_MODEL:-models/base/Qwen3.5-4B}
ASR=${ASR:-gipformer}
ASR_MODEL=${ASR_MODEL:-}
ASR_OUT=outputs/asr_${ASR}_train.jsonl
if [[ "$MIX_ASR" == 1 ]]; then DATA_DIR=outputs/sft_${HOLDOUT}_mix; else DATA_DIR=outputs/sft_${HOLDOUT}_gold; fi
mkdir -p outputs

if pgrep -u "$(id -u)" -f "bin/vllm serve" > /dev/null; then
  echo "a vLLM server of yours is running; stop it first (kill \$VLLM_PID) so training has the whole GPU" >&2
  exit 1
fi

if [[ "$MIX_ASR" == 1 ]]; then
  echo "== 1/3 ASR transcripts of train ($ASR ${ASR_MODEL:-default model})"
  if [[ -s "$ASR_OUT" ]]; then
    echo "cached: $ASR_OUT"
  else
    lifemate transcribe --split train --asr "$ASR" ${ASR_MODEL:+--asr-model "$ASR_MODEL"} --out "$ASR_OUT"
  fi
  lifemate asr-eval --split train --transcripts "$ASR_OUT"

else
  echo "== 1/3 ASR transcripts: skipped (MIX_ASR=0, gold transcripts only)"
fi

echo "== 2/3 SFT data (holdout=$HOLDOUT) -> $DATA_DIR"
if [[ -s "$DATA_DIR/train.jsonl" ]]; then
  echo "cached: $DATA_DIR/train.jsonl"
else
  lifemate build-sft --split train --asr oracle --holdout "$HOLDOUT" --n-apps 10 \
    --augment "$AUGMENT_GOLD" --fewshot-k "$FEWSHOT" --out-dir "$DATA_DIR/gold"
  parts=("$DATA_DIR/gold/train.jsonl")
  if [[ "$MIX_ASR" == 1 ]]; then
    lifemate build-sft --split train --transcripts "$ASR_OUT" --holdout "$HOLDOUT" --n-apps 10 \
      --augment "$AUGMENT_ASR" --fewshot-k "$FEWSHOT" --out-dir "$DATA_DIR/asr"
    parts+=("$DATA_DIR/asr/train.jsonl")
  fi
  cat "${parts[@]}" | shuf --random-source=<(yes) > "$DATA_DIR/train.jsonl"
  if [[ -s "$DATA_DIR/gold/val_ids.txt" ]]; then
    cp "$DATA_DIR/gold/val_ids.txt" "$DATA_DIR/val_ids.txt"
    cp "$DATA_DIR/gold/val.jsonl" "$DATA_DIR/val.jsonl"
  fi
fi
echo "$(wc -l < "$DATA_DIR/train.jsonl") training examples"

echo "== 3/3 LoRA SFT -> checkpoints/$NAME ($EPOCHS epoch)"
python -m lifemate.train.sft_lora --config configs/sft_lora_24gb.yaml \
  train_file="$DATA_DIR/train.jsonl" eval_file=null output_dir="checkpoints/$NAME" \
  num_train_epochs="$EPOCHS" max_length="$MAX_LENGTH" 2>&1 | tee "outputs/$NAME.log"

echo
echo "adapter: checkpoints/$NAME"
if [[ -s "$DATA_DIR/val_ids.txt" ]]; then
  echo "evaluate on validation: VAL=$DATA_DIR/val_ids.txt LORA=checkpoints/$NAME (see README, section 3)"
fi
echo "public_test submission: LORA=checkpoints/$NAME bash scripts/predict_public.sh"
