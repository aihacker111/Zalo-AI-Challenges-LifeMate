#!/usr/bin/env bash
# Fine-tune gipformer1.5-68M (Zipformer RNN-T) on the LifeMate train audio with icefall, then export
# ONNX files that `lifemate transcribe --asr gipformer --asr-model <dir>` loads directly.
#
# Prerequisites
#   1. Any env with a CUDA torch (e.g. conda activate lifemate). Missing k2 / lhotse / icefall are installed
#      on the first run; k2 is matched to the torch already installed, torch itself is never changed.
#   2. ASR manifests from the lifemate environment:
#        lifemate asr-manifest --split train --holdout apps --out-dir outputs/asr_ft     # with a val set
#        lifemate asr-manifest --split train --holdout none --out-dir outputs/asr_full   # all of train
#
# Usage
#   bash scripts/gipformer_finetune.sh                                      # val run: outputs/asr_ft
#   MANIFEST=outputs/asr_full TAG=full bash scripts/gipformer_finetune.sh   # all data, final model
#   STAGE=4 EPOCH=6 bash scripts/gipformer_finetune.sh                      # export another epoch
#
# Stages: 0 download base model, 1 convert checkpoint, 2 lhotse cuts + fbank, 3 train, 4 export ONNX.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:-0}
STOP_STAGE=${STOP_STAGE:-4}
MANIFEST=${MANIFEST:-outputs/asr_ft}
TAG=${TAG:-ft}
EPOCHS=${EPOCHS:-10}
BASE_LR=${BASE_LR:-0.002}
MAX_DURATION=${MAX_DURATION:-600}     # seconds of audio per batch; lower it on OOM
EPOCH=${EPOCH:-$EPOCHS}               # epoch to export
AVG=${AVG:-2}                         # average the model over the last AVG epochs (icefall model_avg)
SPEED_PERTURB=${SPEED_PERTURB:-1}
export ICEFALL_DIR=${ICEFALL_DIR:-$PWD/third_party/icefall}

BASE=models/gipformer/base
DATA=outputs/gipformer/data_$TAG
EXP=checkpoints/gipformer-$TAG
OUT=${OUT:-models/gipformer-$TAG-e$EPOCH}
RECIPE=$ICEFALL_DIR/egs/librispeech/ASR/zipformer
export PYTHONPATH=$ICEFALL_DIR:${PYTHONPATH:-}

# Dependencies, installed into the current env only when missing.
if ! python -c "import k2" 2>/dev/null; then
  echo "== k2 missing: installing the wheel built for this env's torch"
  python recipes/gipformer/install_k2.py
fi
if ! python -c "import lhotse, sentencepiece, kaldialign, pypinyin, typeguard, dill, onnx, onnxruntime, tensorboard" 2>/dev/null; then
  echo "== installing lhotse and icefall's Python dependencies"
  pip install lhotse sentencepiece kaldialign pypinyin typeguard dill tensorboard "onnx>=1.15" "onnxruntime>=1.16.3"
fi
if [[ ! -d "$RECIPE" ]]; then
  echo "== cloning icefall -> $ICEFALL_DIR"
  mkdir -p "$(dirname "$ICEFALL_DIR")"
  git clone --depth 1 https://github.com/k2-fsa/icefall "$ICEFALL_DIR"
fi
if [[ -n "$(pgrep -u "$(id -u)" -f "bin/vllm serve" || true)" ]]; then
  echo "a vLLM server of yours is running and holds GPU memory; stop it first (pgrep -af 'bin/vllm serve')"; exit 1
fi
run() { (( STAGE <= $1 && STOP_STAGE >= $1 )); }

if run 0; then
  echo "== 0/4 base model -> $BASE"
  python - "$BASE" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download("g-group-ai-lab/gipformer1.5-68M-rnnt", local_dir=sys.argv[1])
print("downloaded to", path)
EOF
  ls -la "$BASE"
fi

if run 1; then
  echo "== 1/4 convert model.pt -> $BASE/pretrained.pt"
  python recipes/gipformer/convert_checkpoint.py --src "$BASE/model.pt" --bpe-model "$BASE/bpe.model" \
    --out "$BASE/pretrained.pt"
fi

if run 2; then
  echo "== 2/4 cuts + fbank: $MANIFEST -> $DATA"
  [[ -s "$MANIFEST/train.jsonl" ]] || { echo "missing $MANIFEST/train.jsonl (run lifemate asr-manifest)"; exit 1; }
  python recipes/gipformer/prepare_cuts.py --manifest-dir "$MANIFEST" --bpe-model "$BASE/bpe.model" \
    --out-dir "$DATA" --speed-perturb "$SPEED_PERTURB"
fi

if run 3; then
  echo "== 3/4 fine-tune ($EPOCHS epochs, lr $BASE_LR) -> $EXP (log: $EXP/log/)"
  mkdir -p "$EXP"
  python recipes/gipformer/train_gipformer.py \
    --world-size 1 --num-epochs "$EPOCHS" --start-epoch 1 --exp-dir "$EXP" \
    --manifest-dir "$DATA" --bpe-model "$BASE/bpe.model" \
    --do-finetune 1 --finetune-ckpt "$BASE/pretrained.pt" \
    --base-lr "$BASE_LR" --use-fp16 1 --max-duration "$MAX_DURATION" \
    --enable-musan 0 --enable-spec-aug 1 --num-workers 4 --keep-last-k 3
fi

if run 4; then
  if (( EPOCH > AVG )); then avg_args=(--epoch "$EPOCH" --avg "$AVG" --use-averaged-model 1)
  else avg_args=(--epoch "$EPOCH" --avg 1 --use-averaged-model 0); fi
  suffix="epoch-$EPOCH-avg-${avg_args[3]}"
  echo "== 4/4 export $suffix -> $OUT"
  python "$RECIPE/export-onnx.py" --tokens "$BASE/tokens.txt" --exp-dir "$EXP" "${avg_args[@]}"
  mkdir -p "$OUT"
  for part in encoder decoder joiner; do
    cp "$EXP/$part-$suffix.onnx" "$OUT/$part.onnx"
    [[ -f "$EXP/$part-$suffix.int8.onnx" ]] && cp "$EXP/$part-$suffix.int8.onnx" "$OUT/$part.int8.onnx"
  done
  cp "$BASE/tokens.txt" "$OUT/"
  ls -la "$OUT"
  echo "Evaluate:"
  echo "  ENGINES=\"gipformer gipformer:$OUT\" bash scripts/asr_benchmark.sh"
fi
