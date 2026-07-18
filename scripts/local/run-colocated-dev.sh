#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ENV="${LOCAL_ENV:-$ROOT_DIR/.env.local}"

if [[ -f "$LOCAL_ENV" ]]; then
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
NODE0_PORT="${JF_NODE0_PORT:-19180}"
NODE1_PORT="${JF_NODE1_PORT:-19181}"
DEV_CLUSTER_ID="${JF_DEV_CLUSTER_ID:-${NODE_CLUSTER_ID:-home-lab}-colocated}"
WORK_DIR="${JF_DEV_WORK_DIR:-$ROOT_DIR/.cache/jetsonfabric/colocated-dev}"
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

PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "JetsonFabric colocated development cluster stopped with an error." >&2
    for log_file in "$LOG_DIR"/*.log; do
      [[ -e "$log_file" ]] || continue
      echo "===== $log_file =====" >&2
      tail -n 120 "$log_file" >&2
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

for command in curl jq go cmake head; do
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
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 2 ]]; then
  echo "invalid model layer count: $LAYER_COUNT" >&2
  exit 1
fi
SPLIT_LAYER=$((LAYER_COUNT / 2))

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

start_node() {
  local name=$1
  local port=$2
  local peer_port=$3
  local stage_index=$4
  local layer_start=$5
  local layer_end=$6
  local role=$7

  "$NODE_BIN" \
    --cluster-id "$DEV_CLUSTER_ID" \
    --node-name "$name" \
    --listen "127.0.0.1:$port" \
    --advertise-url "http://127.0.0.1:$port" \
    --data-dir "$WORK_DIR/$name" \
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
    --stage-index "$stage_index" \
    --stage-count 2 \
    --layer-start "$layer_start" \
    --layer-end "$layer_end" \
    --role "$role" \
    --discovery static \
    --seeds "http://127.0.0.1:$peer_port" \
    --discovery-interval 250ms \
    --stale-after 5s \
    --benchmarks "$WORK_DIR/$name-benchmarks.jsonl" \
    --models "$MODEL_REGISTRY" \
    >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
}

start_node jf-dev-stage0 "$NODE0_PORT" "$NODE1_PORT" 0 0 "$SPLIT_LAYER" coordinator
start_node jf-dev-stage1 "$NODE1_PORT" "$NODE0_PORT" 1 "$SPLIT_LAYER" "$LAYER_COUNT" worker

wait_for_url() {
  local url=$1
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "timed out waiting for $url" >&2
  return 1
}

wait_for_two_members() {
  local url=$1
  for _ in $(seq 1 240); do
    if [[ "$(curl -fsS "$url" 2>/dev/null | jq '.members | length' 2>/dev/null || echo 0)" == "2" ]]; then
      return 0
    fi
    sleep 0.25
  done
  echo "timed out waiting for two colocated members" >&2
  return 1
}

NODE0_URL="http://127.0.0.1:$NODE0_PORT"
NODE1_URL="http://127.0.0.1:$NODE1_PORT"
wait_for_url "$NODE0_URL/healthz"
wait_for_url "$NODE1_URL/healthz"
wait_for_two_members "$NODE0_URL/v1/cluster/members"

curl -fsS "$NODE0_URL/v1/routes/preview?model=$MODEL_ID&stage_count=2&allow_colocated_stages=true" \
  | jq -e '.valid == true and .stage_count == 2 and .topology == "colocated"' >/dev/null

cat <<EOF
JetsonFabric colocated development cluster is ready.

Model:       $MODEL_ID
Layers:      [0,$SPLIT_LAYER) -> [$SPLIT_LAYER,$LAYER_COUNT)
Backend:     $RUNTIME_COMPUTE_BACKEND
Node 0:      $NODE0_URL
Node 1:      $NODE1_URL
Logs:        $LOG_DIR

In another terminal:
  make dev-status
  make dev-chat DEV_PROMPT='Explain JetsonFabric in one sentence.'

The base STAGE_COUNT/STAGE_INDEX/LAYER_* values in .env.local are not used by
this launcher; it derives the model layer count and creates two assignments.

Press Ctrl+C to stop both Go nodes and their supervised C++ runtimes.
EOF

while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      exit $?
    fi
  done
  sleep 1
done
