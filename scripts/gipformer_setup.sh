#!/usr/bin/env bash
# One-off environment for fine-tuning gipformer with icefall (Linux, NVIDIA GPU, driver for CUDA 12.8).
# Kept apart from the lifemate/vLLM environment: icefall needs k2, built for one exact torch version.
#
#   bash scripts/gipformer_setup.sh
#   source .venv-gipformer/bin/activate        # before scripts/gipformer_finetune.sh
#
# Versions follow gipformer's own pyproject (torch 2.8.0 + k2 1.24.4, CUDA 12.8).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-gipformer}
ICEFALL_DIR=${ICEFALL_DIR:-third_party/icefall}
PYTHON=${PYTHON:-python3}

if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v uv >/dev/null; then uv venv --python 3.11 "$VENV"; else "$PYTHON" -m venv "$VENV"; fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PIP="pip"
command -v uv >/dev/null && PIP="uv pip"
$PIP install -U pip setuptools wheel
$PIP install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install "k2==1.24.4.dev20250807+cuda12.8.torch2.8.0" -f https://k2-fsa.github.io/k2/cuda.html
$PIP install lhotse sentencepiece kaldialign "pypinyin==0.50.0" tensorboard typeguard dill packaging \
  "onnx>=1.15" "onnxruntime>=1.16.3" soundfile huggingface_hub

if [[ ! -d "$ICEFALL_DIR/.git" ]]; then
  mkdir -p "$(dirname "$ICEFALL_DIR")"
  git clone --depth 1 https://github.com/k2-fsa/icefall "$ICEFALL_DIR"
fi

python - <<'EOF'
import k2, lhotse, torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"k2 {k2.__version__}, lhotse {lhotse.__version__}")
assert torch.cuda.is_available(), "no GPU visible: check the NVIDIA driver (CUDA 12.8 needs driver >= 570)"
EOF
echo "OK. Next: source $VENV/bin/activate && bash scripts/gipformer_finetune.sh"
