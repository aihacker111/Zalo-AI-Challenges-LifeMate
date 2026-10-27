#!/usr/bin/env bash
# One-off conda environment for fine-tuning gipformer with icefall (Linux, NVIDIA GPU, driver for CUDA 12.8).
# A separate env from `lifemate`: k2 is built for exactly torch 2.8.0, while vLLM pins its own torch.
#
#   bash scripts/gipformer_setup.sh                 # creates conda env "gipformer"
#   CONDA_ENV=myenv bash scripts/gipformer_setup.sh # another name (an existing env is reused)
#   conda activate gipformer                        # before scripts/gipformer_finetune.sh
#
# Versions follow gipformer's own pyproject (torch 2.8.0 + k2 1.24.4, CUDA 12.8).
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_ENV=${CONDA_ENV:-gipformer}
ICEFALL_DIR=${ICEFALL_DIR:-third_party/icefall}

command -v conda >/dev/null || { echo "conda not found"; exit 1; }
if [[ "$CONDA_ENV" == lifemate ]]; then
  echo "do not install into 'lifemate': torch 2.8 for k2 would replace the torch vLLM needs"; exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" python=3.11
fi
RUN=(conda run --no-capture-output -n "$CONDA_ENV")

"${RUN[@]}" pip install -U pip setuptools wheel
"${RUN[@]}" pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"${RUN[@]}" pip install "k2==1.24.4.dev20250807+cuda12.8.torch2.8.0" -f https://k2-fsa.github.io/k2/cuda.html
"${RUN[@]}" pip install lhotse sentencepiece kaldialign "pypinyin==0.50.0" tensorboard typeguard dill packaging \
  "onnx>=1.15" "onnxruntime>=1.16.3" soundfile huggingface_hub

if [[ ! -d "$ICEFALL_DIR/.git" ]]; then
  mkdir -p "$(dirname "$ICEFALL_DIR")"
  git clone --depth 1 https://github.com/k2-fsa/icefall "$ICEFALL_DIR"
fi

"${RUN[@]}" python -c '
import k2, lhotse, torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"k2 {k2.__version__}, lhotse {lhotse.__version__}")
assert torch.cuda.is_available(), "no GPU visible: check the NVIDIA driver (CUDA 12.8 needs driver >= 570)"
'
echo "OK. Next: conda activate $CONDA_ENV && bash scripts/gipformer_finetune.sh"
