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
# Settings: HOLDOUT, NAME, EPOCHS, AUGMENT_GOLD, AUGMENT_ASR, FEWSHOT, MAX_LENGTH, ASR_MODEL, BASE_MODEL.
set -euo pipefail
cd "$(dirname "$0")/.."

HOLDOUT=${HOLDOUT:-apps}
NAME=${NAME:-$([[ "$HOLDOUT" == none ]] && echo sft-full || echo sft-v1)}
EPOCHS=${EPOCHS:-1}
AUGMENT_GOLD=${AUGMENT_GOLD:-1}
AUGMENT_ASR=${AUGMENT_ASR:-0}
FEWSHOT=${FEWSHOT:-3}
MAX_LENGTH=${MAX_LENGTH:-12288}
BASE_MODEL=${BASE_MODEL:-models/base/Qwen3.5-4B}
ASR_MODEL=${ASR_MODEL:-models/base/Qwen3-ASR-0.6B}
[[ -e "$ASR_MODEL" || "$ASR_MODEL" != models/* ]] || ASR_MODEL=Qwen/Qwen3-ASR-0.6B
ASR_OUT=outputs/asr0_train.jsonl
DATA_DIR=outputs/sft_${HOLDOUT}
mkdir -p outputs

if ss -ltn 2>/dev/null | grep -qE ':(8011|8000) ' && pgrep -u "$USER" -f "vllm serve" > /dev/null; then
  echo "a vLLM server of yours is running; stop it first (kill \$VLLM_PID) so training has the whole GPU" >&2
  exit 1
fi

echo "== 1/3 ASR transcripts of train ($ASR_MODEL)"
if [[ -s "$ASR_OUT" ]]; then
  echo "cached: $ASR_OUT"
else
  lifemate transcribe --split train --asr qwen --asr-model "$ASR_MODEL" --out "$ASR_OUT"
fi
lifemate asr-eval --split train --transcripts "$ASR_OUT"

echo "== 2/3 SFT data (holdout=$HOLDOUT) -> $DATA_DIR"
if [[ -s "$DATA_DIR/train.jsonl" ]]; then
  echo "cached: $DATA_DIR/train.jsonl"
else
  lifemate build-sft --split train --asr oracle --holdout "$HOLDOUT" --n-apps 10 \
    --augment "$AUGMENT_GOLD" --fewshot-k "$FEWSHOT" --out-dir "$DATA_DIR/gold"
  lifemate build-sft --split train --transcripts "$ASR_OUT" --holdout "$HOLDOUT" --n-apps 10 \
    --augment "$AUGMENT_ASR" --fewshot-k "$FEWSHOT" --out-dir "$DATA_DIR/asr"
  cat "$DATA_DIR/gold/train.jsonl" "$DATA_DIR/asr/train.jsonl" | shuf --random-source=<(yes) > "$DATA_DIR/train.jsonl"
  if [[ -s "$DATA_DIR/asr/val_ids.txt" ]]; then
    cp "$DATA_DIR/asr/val_ids.txt" "$DATA_DIR/val_ids.txt"
    cp "$DATA_DIR/asr/val.jsonl" "$DATA_DIR/val.jsonl"
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
