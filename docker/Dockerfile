# Offline inference image for the final submission.
# Build context = repository root, with the weights placed under models/ beforehand:
#   models/llm/  merged LLM checkpoint (e.g. Qwen3.5-4B + LoRA, see lifemate.train.merge_lora)
#   models/asr/  ASR model: gipformer ONNX files (encoder/decoder/joiner.onnx, tokens.txt) or a Qwen3-ASR checkpoint
#   assets/train/train.jsonl  few-shot pool (training data)
# docker build -f docker/Dockerfile -t lifemate:final .
# docker run --gpus all --network none -v /path/to/private_test:/data lifemate:final

FROM vllm/vllm-openai:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    PYTHONHASHSEED=0

WORKDIR /app
RUN pip install --no-cache-dir "vllm[audio]" sherpa-onnx sherpa-onnx-core soundfile huggingface_hub

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e .

COPY models /models
COPY assets /app/assets
COPY docker/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
