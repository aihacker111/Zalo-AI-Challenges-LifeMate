#!/usr/bin/env bash
# Build SFT data (leave-10-apps-out validation, 2 renamed copies per sample) and train LoRA on one 24 GB GPU.
#   bash scripts/train_sft_24gb.sh ../train+public_test [outputs/transcripts_train.jsonl]
set -euo pipefail
DATA=${1:-../train+public_test}
TRANSCRIPTS=${2:-}

ASR_ARGS=(--asr oracle)
if [[ -n "$TRANSCRIPTS" ]]; then ASR_ARGS=(--transcripts "$TRANSCRIPTS"); fi

lifemate build-sft --data "$DATA" --split train "${ASR_ARGS[@]}" \
  --holdout apps --n-apps 10 --augment 2 --fewshot-k 3 --out-dir outputs/sft
python -m lifemate.train.sft_lora --config configs/sft_lora_24gb.yaml
