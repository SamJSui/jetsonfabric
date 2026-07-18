#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ENV="${LOCAL_ENV:-$ROOT_DIR/.env.local}"

# `make dev-up` already parsed .env.local and passes the resolved settings.
# Source it only when this script is invoked directly.
if [[ -z "${MODEL_PATH:-}" && -f "$LOCAL_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_ENV"
  set +a
fi

MODEL_ID="${MODEL_ID:-${MODEL:-qwen2.5-coder-1.5b-q4}}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to a local GGUF file}"
RUNTIME_COMPUTE_BACKEND="${RUNTIME_COMPUTE_BACKEND:-cuda}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-runtime/build}"
RUNTIME_BIN="${RUNTIME_BIN:-dist/jetsonfabric-runtime-worker}"
NODE_BIN="${NODE_BIN:-dist/jetsonfabric-node}"
NODE_PORT="${JF_NODE0_PORT:-19180}"
DEV_CLUSTER_ID="${JF_DEV_CLUSTER_ID:-${NODE_CLUSTER_ID:-home-lab}-dev}"
WORK_DIR="${JF_DEV_WORK_DIR:-$ROOT_DIR/.cache/jetsonfabric/dev}"
LOG_DIR="$WORK_DIR/logs"

absolute_from_root() {
  local value=$1
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$value"
  fi
}

MODEL_PATH="$(absolute_from_root "$MODEL_PATH")"
RUNTIME_BUILD_DIR="$(absolute_from_root "$RUNTIME_BUILD_DIR")"
RUNTIME_BIN="$(absolute_from_root "$RUNTIME_BIN")"
NODE_BIN="$(absolute_from_root "$NODE_BIN")"

rm -rf "$WORK_DIR"
mkdir -p "$LOG_DIR"

NODE_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$NODE_PID" ]]; then
    kill "$NODE_PID" 2>/dev/null || true
    wait "$NODE_PID" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "JetsonFabric development node stopped with an error." >&2
    for log_file in "$LOG_DIR"/*.log; do
      [[ -e "$log_file" ]] || continue
      echo "===== $log_file =====" >&2
      tail -n 160 "$log_file" >&2
    done
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}
for command in curl jq go cmake make head seq; do
  require_command "$command"
done

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ "$(head -c 4 "$MODEL_PATH")" != "GGUF" ]]; then
  echo "model is not a GGUF file: $MODEL_PATH" >&2
  exit 2
fi

cd "$ROOT_DIR"
if [[ "${JF_SKIP_BUILD:-false}" != "true" ]]; then
  if [[ "$RUNTIME_COMPUTE_BACKEND" == "cuda" ]]; then
    make runtime-cuda \
      RUNTIME_BUILD_DIR="$RUNTIME_BUILD_DIR" \
      RUNTIME_BIN="$RUNTIME_BIN" \
      RUNTIME_BUILD_JOBS="${RUNTIME_BUILD_JOBS:-1}" \
      RUNTIME_CUDA_ARCH="${RUNTIME_CUDA_ARCH:-87}" \
      CUDA_NVCC="${CUDA_NVCC:-/usr/local/cuda/bin/nvcc}"
  else
    make runtime \
      RUNTIME_BUILD_DIR="$RUNTIME_BUILD_DIR" \
      RUNTIME_BIN="$RUNTIME_BIN" \
      RUNTIME_BUILD_JOBS="${RUNTIME_BUILD_JOBS:-1}"
  fi
  mkdir -p "$(dirname "$NODE_BIN")"
  go build -buildvcs=false -o "$NODE_BIN" ./cmd/jetsonfabric-node
fi

STAGE_TEST_BIN="$RUNTIME_BUILD_DIR/jetsonfabric-llama-stage-test"
for binary in "$RUNTIME_BIN" "$NODE_BIN" "$STAGE_TEST_BIN"; do
  if [[ ! -x "$binary" ]]; then
    echo "required binary is missing: $binary" >&2
    echo "Run without JF_SKIP_BUILD=true to build it." >&2
    exit 2
  fi
done

LAYER_COUNT="$(CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" --print-layer-count)"
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 1 ]]; then
  echo "invalid model layer count: $LAYER_COUNT" >&2
  exit 1
fi

MODEL_REGISTRY="$WORK_DIR/models.json"
cat >"$MODEL_REGISTRY" <<JSON
{
  "models": [{
    "id": "$MODEL_ID",
    "family": "llm",
    "supported_engines": ["llama.cpp"],
    "layer_count": $LAYER_COUNT,
    "min_memory_gb": 0,
    "preferred_compute": null,
    "placement_modes": ["pipeline_parallel"]
  }]
}
JSON

NODE_URL="http://127.0.0.1:$NODE_PORT"
"$NODE_BIN" \
  --cluster-id "$DEV_CLUSTER_ID" \
  --node-name jf-dev \
  --listen "127.0.0.1:$NODE_PORT" \
  --advertise-url "$NODE_URL" \
  --data-dir "$WORK_DIR/node" \
  --runtime-url auto \
  --runtime-bin "$RUNTIME_BIN" \
  --runtime-listen 127.0.0.1:0 \
  --runtime-compute-backend "$RUNTIME_COMPUTE_BACKEND" \
  --runtime-mode pipeline_parallel \
  --runtime-ctx-size "${RUNTIME_CTX_SIZE:-4096}" \
  --runtime-n-gpu-layers "${RUNTIME_N_GPU_LAYERS:-999}" \
  --runtime-threads "${RUNTIME_THREADS:-0}" \
  --engine "${NODE_ENGINE:-llama.cpp}" \
  --model "$MODEL_ID" \
  --model-path "$MODEL_PATH" \
  --stage-index 0 \
  --stage-count 1 \
  --layer-start 0 \
  --layer-end "$LAYER_COUNT" \
  --role coordinator \
  --discovery none \
  --discovery-interval 1s \
  --stale-after 30s \
  --benchmarks "$WORK_DIR/benchmarks.jsonl" \
  --models "$MODEL_REGISTRY" \
  >"$LOG_DIR/jf-dev.log" 2>&1 &
NODE_PID=$!

wait_for_url() {
  local url=$1
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
      wait "$NODE_PID"
      return $?
    fi
    sleep 0.25
  done
  echo "timed out waiting for $url" >&2
  return 1
}

wait_for_url "$NODE_URL/healthz"

SMOKE_RESPONSE="$WORK_DIR/readiness-chat.json"
SMOKE_STATUS="$(curl -sS -o "$SMOKE_RESPONSE" -w '%{http_code}' \
  -X POST "$NODE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    '{model:$model,messages:[{role:"user",content:"Say hello."}],max_tokens:1}')")"
if [[ "$SMOKE_STATUS" != "200" ]]; then
  echo "single-node inference readiness check returned HTTP $SMOKE_STATUS" >&2
  jq . "$SMOKE_RESPONSE" >&2 2>/dev/null || cat "$SMOKE_RESPONSE" >&2
  exit 1
fi
jq -e '.object == "chat.completion" and (.choices | length) == 1' "$SMOKE_RESPONSE" >/dev/null

cat <<EOF
JetsonFabric development node is ready.

Model:       $MODEL_ID
Layers:      [0,$LAYER_COUNT)
Pipeline:    stage 0 of 1
Backend:     $RUNTIME_COMPUTE_BACKEND
Node:        $NODE_URL
Logs:        $LOG_DIR

In another terminal:
  make dev-status
  make dev-chat DEV_PROMPT='Explain JetsonFabric in one sentence.'

Press Ctrl+C to stop the Go node and its supervised C++ runtime.
EOF

while kill -0 "$NODE_PID" 2>/dev/null; do
  sleep 1
done
wait "$NODE_PID"
