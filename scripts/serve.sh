#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server in the background and wait until it is ready.
# Fails (exit 1) with the end of the server log if vLLM dies during start-up.
#   source scripts/serve.sh models/base/Qwen3.5-4B base 8011 0.85 [extra vllm args...]
#   -> exports VLLM_PID and BASE_URL; stop it later with: kill $VLLM_PID
MODEL=${1:?model path}
NAME=${2:-base}
PORT=${3:-8000}
MEM=${4:-0.85}
shift $(( $# < 4 ? $# : 4 ))
LOG=${VLLM_LOG:-outputs/vllm_${PORT}.log}
mkdir -p "$(dirname "$LOG")"

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port $PORT is already in use; choose another one" >&2
  return 1 2>/dev/null || exit 1
fi

vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" --max-model-len 16384 \
  --enable-prefix-caching --gpu-memory-utilization "$MEM" --seed 0 "$@" > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM pid $VLLM_PID, log $LOG"
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during start-up; last lines of $LOG:" >&2
    tail -30 "$LOG" >&2
    return 1 2>/dev/null || exit 1
  fi
  sleep 5
done
export VLLM_PID BASE_URL="http://localhost:$PORT/v1"
echo "ready: $BASE_URL (model name: $NAME)"
