#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GGUF to validate}"
MODEL_ID="${MODEL_ID:-qwen2.5-coder-1.5b-q4}"
DEPLOYMENT_ID="${JF_DEPLOYMENT_ID:-lifecycle-deployment-1}"
RAW_PROMPT="${JF_RAW_PROMPT:-Once upon a time}"
MAX_TOKENS="${JF_MAX_TOKENS:-2}"
NODE_PORT="${JF_NODE0_PORT:-19280}"
RUNTIME_PORT="${JF_RUNTIME0_PORT:-19290}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-lifecycle-cpu}"
RUNTIME_BIN="${RUNTIME_BIN:-$ROOT_DIR/dist/jetsonfabric-runtime-worker-lifecycle-cpu}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/dist/jetsonfabric-node}"
WORK_DIR="$(mktemp -d)"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

RUNTIME_PID=""
NODE_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$NODE_PID" ]]; then
    kill "$NODE_PID" 2>/dev/null || true
    wait "$NODE_PID" 2>/dev/null || true
  fi
  if [[ -n "$RUNTIME_PID" ]]; then
    kill "$RUNTIME_PID" 2>/dev/null || true
    wait "$RUNTIME_PID" 2>/dev/null || true
  fi
  if [[ $status -ne 0 ]]; then
    echo "Runtime deployment lifecycle validation failed." >&2
    for log_file in "$LOG_DIR"/*.log; do
      [[ -e "$log_file" ]] || continue
      echo "===== $log_file =====" >&2
      tail -n 200 "$log_file" >&2
    done
  fi
  rm -rf "$WORK_DIR"
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
for command in curl jq go cmake head seq; do
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
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"

cd "$ROOT_DIR"
if [[ "${JF_SKIP_BUILD:-false}" != "true" ]]; then
  make runtime \
    RUNTIME_BUILD_DIR="$RUNTIME_BUILD_DIR" \
    RUNTIME_BIN="$RUNTIME_BIN" \
    RUNTIME_BUILD_JOBS="${RUNTIME_BUILD_JOBS:-1}"
  mkdir -p "$(dirname "$NODE_BIN")"
  go build -buildvcs=false -o "$NODE_BIN" ./cmd/jetsonfabric-node
fi

STAGE_TEST_BIN="$RUNTIME_BUILD_DIR/jetsonfabric-llama-stage-test"
for binary in "$RUNTIME_BIN" "$NODE_BIN" "$STAGE_TEST_BIN"; do
  if [[ ! -x "$binary" ]]; then
    echo "required binary is missing: $binary" >&2
    exit 2
  fi
done

LAYER_COUNT="$(CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" --print-layer-count)"
BASELINE_TOKENS="$(CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" --baseline-tokens)"
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 1 ]]; then
  echo "invalid layer count: $LAYER_COUNT" >&2
  exit 1
fi
if ! jq -e 'type == "array" and length == 2 and all(.[]; type == "number")' <<<"$BASELINE_TOKENS" >/dev/null; then
  echo "invalid baseline tokens: $BASELINE_TOKENS" >&2
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

RUNTIME_URL="http://127.0.0.1:$RUNTIME_PORT"
NODE_URL="http://127.0.0.1:$NODE_PORT"

"$RUNTIME_BIN" \
  --listen "127.0.0.1:$RUNTIME_PORT" \
  --node-name lifecycle-runtime \
  --idle \
  --engine llama.cpp \
  --compute-backend cpu \
  --mode pipeline_parallel \
  --ctx-size "${JF_CTX_SIZE:-256}" \
  --n-gpu-layers 0 \
  --threads "${JF_RUNTIME_THREADS:-2}" \
  >"$LOG_DIR/runtime.log" 2>&1 &
RUNTIME_PID=$!

wait_for_url() {
  local url=$1
  local pid=$2
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      return $?
    fi
    sleep 0.25
  done
  echo "timed out waiting for $url" >&2
  return 1
}

wait_for_url "$RUNTIME_URL/v1/deployment" "$RUNTIME_PID"
IDLE_FILE="$WORK_DIR/idle.json"
curl -fsS "$RUNTIME_URL/v1/deployment" >"$IDLE_FILE"
jq -e '
  .resident == false and
  .active == false and
  .state == "idle" and
  .deployment == null
' "$IDLE_FILE" >/dev/null

LOAD_FILE="$WORK_DIR/load.json"
LOAD_CODE="$(curl -sS -o "$LOAD_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/load" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg deployment "$DEPLOYMENT_ID" \
    --arg model "$MODEL_ID" \
    --arg path "$MODEL_PATH" \
    --argjson layer_end "$LAYER_COUNT" \
    --argjson ctx_size "${JF_CTX_SIZE:-256}" \
    --argjson threads "${JF_RUNTIME_THREADS:-2}" \
    '{deployment_id:$deployment,model_id:$model,engine:"llama.cpp",compute_backend:"cpu",model_path:$path,ctx_size:$ctx_size,n_gpu_layers:0,threads:$threads,mode:"pipeline_parallel",stage_index:0,stage_count:1,layer_start:0,layer_end:$layer_end}')")"
if [[ "$LOAD_CODE" != "200" ]]; then
  echo "runtime load returned HTTP $LOAD_CODE" >&2
  jq . "$LOAD_FILE" >&2 2>/dev/null || cat "$LOAD_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" --arg model "$MODEL_ID" '
  .loaded == true and
  .resident == true and
  .active == false and
  .state == "ready" and
  .deployment.deployment_id == $deployment and
  .deployment.model_id == $model
' "$LOAD_FILE" >/dev/null

READY_STATUS="$WORK_DIR/ready-status.json"
curl -fsS "$RUNTIME_URL/v1/deployment" >"$READY_STATUS"
jq -e --arg deployment "$DEPLOYMENT_ID" '
  .resident == true and
  .active == false and
  .state == "ready" and
  .deployment.deployment_id == $deployment
' "$READY_STATUS" >/dev/null

ACTIVATE_FILE="$WORK_DIR/activate.json"
ACTIVATE_CODE="$(curl -sS -o "$ACTIVATE_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/activate" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc --arg deployment "$DEPLOYMENT_ID" '{deployment_id:$deployment}')")"
if [[ "$ACTIVATE_CODE" != "200" ]]; then
  echo "runtime activation returned HTTP $ACTIVATE_CODE" >&2
  jq . "$ACTIVATE_FILE" >&2 2>/dev/null || cat "$ACTIVATE_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" '
  .activated == true and
  .resident == true and
  .active == true and
  .state == "active" and
  .deployment.deployment_id == $deployment
' "$ACTIVATE_FILE" >/dev/null

"$NODE_BIN" \
  --cluster-id lifecycle-integration \
  --node-name lifecycle-node \
  --listen "127.0.0.1:$NODE_PORT" \
  --advertise-url "$NODE_URL" \
  --data-dir "$WORK_DIR/node" \
  --runtime-url "$RUNTIME_URL" \
  --runtime-compute-backend cpu \
  --runtime-mode pipeline_parallel \
  --runtime-ctx-size "${JF_CTX_SIZE:-256}" \
  --runtime-n-gpu-layers 0 \
  --runtime-threads "${JF_RUNTIME_THREADS:-2}" \
  --engine llama.cpp \
  --model "$MODEL_ID" \
  --model-path "$MODEL_PATH" \
  --stage-index 0 \
  --stage-count 1 \
  --layer-start 0 \
  --layer-end "$LAYER_COUNT" \
  --role coordinator \
  --discovery none \
  --discovery-interval 250ms \
  --stale-after 5s \
  --benchmarks "$WORK_DIR/benchmarks.jsonl" \
  --models "$MODEL_REGISTRY" \
  >"$LOG_DIR/node.log" 2>&1 &
NODE_PID=$!

wait_for_url "$NODE_URL/healthz" "$NODE_PID"

INFERENCE_FILE="$WORK_DIR/inference.json"
INFERENCE_CODE="$(curl -sS -o "$INFERENCE_FILE" -w '%{http_code}' \
  -X POST "$NODE_URL/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg payload "$RAW_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{request_id:"runtime-lifecycle-integration",model:$model,payload:$payload,max_tokens:$max_tokens,stage_count:1}')")"
if [[ "$INFERENCE_CODE" != "200" ]]; then
  echo "lifecycle inference returned HTTP $INFERENCE_CODE" >&2
  jq . "$INFERENCE_FILE" >&2 2>/dev/null || cat "$INFERENCE_FILE" >&2
  exit 1
fi
jq -e --argjson baseline "$BASELINE_TOKENS" '
  .plan.valid == true and
  .plan.stage_count == 1 and
  .result.payload_kind == "sampled_token" and
  .result.sampled_tokens == $baseline and
  .result.prompt_tokens > 0
' "$INFERENCE_FILE" >/dev/null

UNLOAD_FILE="$WORK_DIR/unload.json"
UNLOAD_CODE="$(curl -sS -o "$UNLOAD_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/unload" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc --arg deployment "$DEPLOYMENT_ID" '{deployment_id:$deployment}')")"
if [[ "$UNLOAD_CODE" != "200" ]]; then
  echo "runtime unload returned HTTP $UNLOAD_CODE" >&2
  jq . "$UNLOAD_FILE" >&2 2>/dev/null || cat "$UNLOAD_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" '
  .unloaded == true and
  .resident == false and
  .active == false and
  .state == "idle" and
  .deployment.deployment_id == $deployment
' "$UNLOAD_FILE" >/dev/null

FINAL_STATUS="$WORK_DIR/final-status.json"
curl -fsS "$RUNTIME_URL/v1/deployment" >"$FINAL_STATUS"
jq -e '
  .resident == false and
  .active == false and
  .state == "idle" and
  .deployment == null
' "$FINAL_STATUS" >/dev/null

echo "Runtime deployment lifecycle validation passed"
echo "Lifecycle: idle -> load -> ready -> activate -> infer -> unload -> idle"
echo "Model: $MODEL_ID"
echo "Layers: [0,$LAYER_COUNT)"
