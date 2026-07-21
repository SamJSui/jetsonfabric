#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GGUF to validate}"
MODEL_ID="${MODEL_ID:-qwen2.5-coder-1.5b-q4}"
DEPLOYMENT_ID="${JF_DEPLOYMENT_ID:-lifecycle-deployment-1}"
DEPLOYMENT_EPOCH="${JF_DEPLOYMENT_EPOCH:-1}"
NODE_NAME="${JF_NODE_NAME:-lifecycle-node}"
RAW_PROMPT="${JF_RAW_PROMPT:-Once upon a time}"
MAX_TOKENS="${JF_MAX_TOKENS:-4}"
EXPECTED_TOKENS="${JF_EXPECTED_TOKENS:-}"
NODE_PORT="${JF_NODE0_PORT:-19280}"
RUNTIME_PORT="${JF_RUNTIME0_PORT:-19290}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-lifecycle-cpu}"
RUNTIME_BIN="${RUNTIME_BIN:-$ROOT_DIR/dist/jetsonfabric-runtime-worker-lifecycle-cpu}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/dist/jetsonfabric-node}"
CLUSTER_TOKEN="${JF_CLUSTER_TOKEN:-jetsonfabric-lifecycle-integration-token}"
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
for command in curl jq go cmake head seq sha256sum awk; do
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
MODEL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"

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
BASELINE_TOKENS="$(
  CI_MODEL_PATH="$MODEL_PATH" \
  JF_BASELINE_PROMPT="$RAW_PROMPT" \
  JF_BASELINE_MAX_TOKENS="$MAX_TOKENS" \
  "$STAGE_TEST_BIN" --baseline-tokens
)"
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 1 ]]; then
  echo "invalid layer count: $LAYER_COUNT" >&2
  exit 1
fi
if ! jq -e --argjson count "$MAX_TOKENS" 'type == "array" and length == $count and all(.[]; type == "number")' <<<"$BASELINE_TOKENS" >/dev/null; then
  echo "invalid baseline tokens: $BASELINE_TOKENS" >&2
  exit 1
fi
if [[ -n "$EXPECTED_TOKENS" ]] && ! jq -e --argjson actual "$BASELINE_TOKENS" --argjson expected "$EXPECTED_TOKENS" '$actual == $expected' <<<null >/dev/null; then
  echo "baseline tokens do not match JF_EXPECTED_TOKENS: expected=$EXPECTED_TOKENS actual=$BASELINE_TOKENS" >&2
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
  --node-name "$NODE_NAME" \
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
  .deployment == null and
  .model_memory == null
' "$IDLE_FILE" >/dev/null

LOAD_FILE="$WORK_DIR/load.json"
LOAD_CODE="$(curl -sS -o "$LOAD_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/load" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg deployment "$DEPLOYMENT_ID" \
    --argjson epoch "$DEPLOYMENT_EPOCH" \
    --arg model "$MODEL_ID" \
    --arg model_sha256 "$MODEL_SHA256" \
    --arg path "$MODEL_PATH" \
    --argjson layer_end "$LAYER_COUNT" \
    --argjson ctx_size "${JF_CTX_SIZE:-256}" \
    --argjson threads "${JF_RUNTIME_THREADS:-2}" \
    '{deployment_id:$deployment,epoch:$epoch,model_id:$model,model_sha256:$model_sha256,engine:"llama.cpp",compute_backend:"cpu",model_path:$path,ctx_size:$ctx_size,n_gpu_layers:0,threads:$threads,mode:"pipeline_parallel",stage_index:0,stage_count:1,layer_start:0,layer_end:$layer_end}')")"
if [[ "$LOAD_CODE" != "200" ]]; then
  echo "runtime load returned HTTP $LOAD_CODE" >&2
  jq . "$LOAD_FILE" >&2 2>/dev/null || cat "$LOAD_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model "$MODEL_ID" --arg model_sha256 "$MODEL_SHA256" --argjson layer_count "$LAYER_COUNT" '
  .loaded == true and
  .resident == true and
  .active == false and
  .state == "ready" and
  .deployment.deployment_id == $deployment and
  .deployment.epoch == $epoch and
  .deployment.model_id == $model and
  .deployment.model_sha256 == $model_sha256 and
  .model_memory.layer_start == 0 and
  .model_memory.layer_end == $layer_count and
  .model_memory.layer_count == $layer_count and
  .model_memory.resident_weight_bytes > 0 and
  .model_memory.resident_weight_bytes == .model_memory.total_weight_bytes and
  .model_memory.resident_tensor_count > 0 and
  .model_memory.partitioned == false and
  .model_memory.pinned == false
' "$LOAD_FILE" >/dev/null

READY_STATUS="$WORK_DIR/ready-status.json"
curl -fsS "$RUNTIME_URL/v1/deployment" >"$READY_STATUS"
jq -e --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model_sha256 "$MODEL_SHA256" '
  .resident == true and
  .active == false and
  .state == "ready" and
  .deployment.deployment_id == $deployment and
  .deployment.epoch == $epoch and
  .deployment.model_sha256 == $model_sha256 and
  .model_memory.pinned == false
' "$READY_STATUS" >/dev/null

ACTIVATE_FILE="$WORK_DIR/activate.json"
ACTIVATE_CODE="$(curl -sS -o "$ACTIVATE_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/activate" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model "$MODEL_ID" --arg model_sha256 "$MODEL_SHA256" '{deployment_id:$deployment,epoch:$epoch,model_id:$model,model_sha256:$model_sha256}')")"
if [[ "$ACTIVATE_CODE" != "200" ]]; then
  echo "runtime activation returned HTTP $ACTIVATE_CODE" >&2
  jq . "$ACTIVATE_FILE" >&2 2>/dev/null || cat "$ACTIVATE_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model_sha256 "$MODEL_SHA256" '
  .activated == true and
  .resident == true and
  .active == true and
  .state == "active" and
  .deployment.deployment_id == $deployment and
  .deployment.epoch == $epoch and
  .deployment.model_sha256 == $model_sha256 and
  .model_memory.pinned == true
' "$ACTIVATE_FILE" >/dev/null

JETSONFABRIC_CLUSTER_TOKEN="$CLUSTER_TOKEN" "$NODE_BIN" \
  --cluster-id lifecycle-integration \
  --node-name "$NODE_NAME" \
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
COORDINATOR_NODE_ID="$(curl -fsS "$NODE_URL/healthz" | jq -r '.node_id')"
if [[ -z "$COORDINATOR_NODE_ID" || "$COORDINATOR_NODE_ID" == "null" ]]; then
  echo "coordinator node ID is unavailable" >&2
  exit 1
fi

INFERENCE_FILE="$WORK_DIR/inference.jsonl"
INFERENCE_CODE="$(curl -sS -o "$INFERENCE_FILE" -w '%{http_code}' \
  -X POST "$NODE_URL/v1/runtime/generate" \
  -H 'Content-Type: application/json' \
  -H "X-JetsonFabric-Coordinator-Node-ID: $COORDINATOR_NODE_ID" \
  -H "X-JetsonFabric-Cluster-Token: $CLUSTER_TOKEN" \
  --data-binary "$(jq -nc \
    --arg deployment "$DEPLOYMENT_ID" \
    --argjson epoch "$DEPLOYMENT_EPOCH" \
    --arg model "$MODEL_ID" \
    --arg model_sha256 "$MODEL_SHA256" \
    --arg payload "$RAW_PROMPT" \
    --arg node_id "$COORDINATOR_NODE_ID" \
    --arg node_name "$NODE_NAME" \
    --arg node_url "$NODE_URL" \
    --argjson layer_end "$LAYER_COUNT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{request_id:"runtime-lifecycle-integration",session_id:"runtime-lifecycle-integration",model_id:$model,prompt:$payload,max_tokens:$max_tokens,deployment:{deployment_id:$deployment,epoch:$epoch,model_id:$model,model_sha256:$model_sha256},stages:[{stage_index:0,stage_count:1,node_id:$node_id,node_name:$node_name,api_url:$node_url,layer_start:0,layer_end:$layer_end}]}')")"
if [[ "$INFERENCE_CODE" != "200" ]]; then
  echo "lifecycle inference returned HTTP $INFERENCE_CODE" >&2
  jq -s . "$INFERENCE_FILE" >&2 2>/dev/null || cat "$INFERENCE_FILE" >&2
  exit 1
fi
jq -s -e --argjson baseline "$BASELINE_TOKENS" '
  ([.[] | select(.type == "done")]) as $done |
  length == (($baseline | length) + 1) and
  all(.[]; .type == "token" or .type == "done") and
  ([.[] | select(.type == "token") | .token] == $baseline) and
  ($done | length) == 1 and
  $done[0].sampled_tokens == $baseline and
  $done[0].stage_calls == ($baseline | length) and
  $done[0].remote_stage_calls == 0 and
  $done[0].prompt_tokens > 0
' "$INFERENCE_FILE" >/dev/null

DRAIN_FILE="$WORK_DIR/drain.json"
DRAIN_CODE="$(curl -sS -o "$DRAIN_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/drain" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model "$MODEL_ID" --arg model_sha256 "$MODEL_SHA256" '{deployment_id:$deployment,epoch:$epoch,model_id:$model,model_sha256:$model_sha256}')")"
if [[ "$DRAIN_CODE" != "200" ]]; then
  echo "runtime drain returned HTTP $DRAIN_CODE" >&2
  jq . "$DRAIN_FILE" >&2 2>/dev/null || cat "$DRAIN_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model_sha256 "$MODEL_SHA256" '
  .drained == true and
  .resident == true and
  .active == true and
  .state == "draining" and
  .deployment.deployment_id == $deployment and
  .deployment.epoch == $epoch and
  .deployment.model_sha256 == $model_sha256 and
  .model_memory.pinned == true
' "$DRAIN_FILE" >/dev/null

UNLOAD_FILE="$WORK_DIR/unload.json"
UNLOAD_CODE="$(curl -sS -o "$UNLOAD_FILE" -w '%{http_code}' \
  -X POST "$RUNTIME_URL/v1/deployment/unload" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model "$MODEL_ID" --arg model_sha256 "$MODEL_SHA256" '{deployment_id:$deployment,epoch:$epoch,model_id:$model,model_sha256:$model_sha256}')")"
if [[ "$UNLOAD_CODE" != "200" ]]; then
  echo "runtime unload returned HTTP $UNLOAD_CODE" >&2
  jq . "$UNLOAD_FILE" >&2 2>/dev/null || cat "$UNLOAD_FILE" >&2
  exit 1
fi
jq -e --arg deployment "$DEPLOYMENT_ID" --argjson epoch "$DEPLOYMENT_EPOCH" --arg model_sha256 "$MODEL_SHA256" '
  .unloaded == true and
  .resident == false and
  .active == false and
  .state == "idle" and
  .deployment.deployment_id == $deployment and
  .deployment.epoch == $epoch and
  .deployment.model_sha256 == $model_sha256 and
  .model_memory == null
' "$UNLOAD_FILE" >/dev/null

FINAL_STATUS="$WORK_DIR/final-status.json"
curl -fsS "$RUNTIME_URL/v1/deployment" >"$FINAL_STATUS"
jq -e '
  .resident == false and
  .active == false and
  .state == "idle" and
  .deployment == null and
  .model_memory == null
' "$FINAL_STATUS" >/dev/null

echo "Runtime deployment lifecycle validation passed"
echo "Lifecycle: idle -> load -> ready -> activate -> infer -> drain -> unload -> idle"
echo "Model: $MODEL_ID"
echo "Layers: [0,$LAYER_COUNT)"
