#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_A_PATH="${MODEL_A_PATH:?MODEL_A_PATH must point to the first GGUF}"
MODEL_B_PATH="${MODEL_B_PATH:?MODEL_B_PATH must point to the second GGUF}"
MODEL_A_ID="${MODEL_A_ID:-ci-tiny-llama-q4}"
MODEL_B_ID="${MODEL_B_ID:-ci-tiny-llama-q8}"
DEPLOYMENT_A="${JF_DEPLOYMENT_A:-coordinator-deployment-a}"
DEPLOYMENT_B="${JF_DEPLOYMENT_B:-coordinator-deployment-b}"
RAW_PROMPT="${JF_RAW_PROMPT:-Once upon a time}"
MAX_TOKENS="${JF_MAX_TOKENS:-4}"
EXPECTED_TOKENS_A="${JF_EXPECTED_TOKENS_A:-}"
EXPECTED_TOKENS_B="${JF_EXPECTED_TOKENS_B:-}"
NODE0_PORT="${JF_NODE0_PORT:-19380}"
NODE1_PORT="${JF_NODE1_PORT:-19381}"
RUNTIME0_PORT="${JF_RUNTIME0_PORT:-19390}"
RUNTIME1_PORT="${JF_RUNTIME1_PORT:-19391}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-switch-cpu}"
RUNTIME_BIN="${RUNTIME_BIN:-$ROOT_DIR/dist/jetsonfabric-runtime-worker-switch-cpu}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/dist/jetsonfabric-node}"
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-unknown}"
RUNTIME_REVISION="${JF_RUNTIME_REVISION:-milestone-6-ci}"
CLUSTER_TOKEN="${JF_CLUSTER_TOKEN:-jetsonfabric-integration-token}"
WORK_DIR="$(mktemp -d)"
LOG_DIR="$WORK_DIR/logs"
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
    echo "Coordinator model switch validation failed." >&2
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

for command in curl jq sha256sum awk go cmake head seq; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required command not found: $command" >&2
    exit 2
  }
done

for model_path in "$MODEL_A_PATH" "$MODEL_B_PATH"; do
  [[ -f "$model_path" ]] || { echo "model does not exist: $model_path" >&2; exit 2; }
  [[ "$(head -c 4 "$model_path")" == "GGUF" ]] || { echo "model is not GGUF: $model_path" >&2; exit 2; }
done
MODEL_A_PATH="$(cd "$(dirname "$MODEL_A_PATH")" && pwd)/$(basename "$MODEL_A_PATH")"
MODEL_B_PATH="$(cd "$(dirname "$MODEL_B_PATH")" && pwd)/$(basename "$MODEL_B_PATH")"
MODEL_A_SHA="$(sha256sum "$MODEL_A_PATH" | awk '{print $1}')"
MODEL_B_SHA="$(sha256sum "$MODEL_B_PATH" | awk '{print $1}')"
[[ "$MODEL_A_SHA" != "$MODEL_B_SHA" ]] || { echo "model switch test requires distinct artifact hashes" >&2; exit 2; }

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
  [[ -x "$binary" ]] || { echo "required binary is missing: $binary" >&2; exit 2; }
done

LAYER_COUNT_A="$(CI_MODEL_PATH="$MODEL_A_PATH" "$STAGE_TEST_BIN" --print-layer-count)"
LAYER_COUNT_B="$(CI_MODEL_PATH="$MODEL_B_PATH" "$STAGE_TEST_BIN" --print-layer-count)"
BASELINE_A="$(
  CI_MODEL_PATH="$MODEL_A_PATH" \
  JF_BASELINE_PROMPT="$RAW_PROMPT" \
  JF_BASELINE_MAX_TOKENS="$MAX_TOKENS" \
  "$STAGE_TEST_BIN" --baseline-tokens
)"
BASELINE_B="$(
  CI_MODEL_PATH="$MODEL_B_PATH" \
  JF_BASELINE_PROMPT="$RAW_PROMPT" \
  JF_BASELINE_MAX_TOKENS="$MAX_TOKENS" \
  "$STAGE_TEST_BIN" --baseline-tokens
)"
[[ "$LAYER_COUNT_A" =~ ^[0-9]+$ && "$LAYER_COUNT_A" -ge 2 ]] || { echo "invalid model A layer count" >&2; exit 1; }
[[ "$LAYER_COUNT_B" == "$LAYER_COUNT_A" ]] || { echo "model layer counts differ: $LAYER_COUNT_A vs $LAYER_COUNT_B" >&2; exit 1; }
for baseline in "$BASELINE_A" "$BASELINE_B"; do
  jq -e --argjson count "$MAX_TOKENS" 'type == "array" and length == $count and all(.[]; type == "number")' <<<"$baseline" >/dev/null
done
if [[ -n "$EXPECTED_TOKENS_A" ]] && ! jq -e --argjson actual "$BASELINE_A" --argjson expected "$EXPECTED_TOKENS_A" '$actual == $expected' <<<null >/dev/null; then
  echo "model A baseline tokens changed: expected=$EXPECTED_TOKENS_A actual=$BASELINE_A" >&2
  exit 1
fi
if [[ -n "$EXPECTED_TOKENS_B" ]] && ! jq -e --argjson actual "$BASELINE_B" --argjson expected "$EXPECTED_TOKENS_B" '$actual == $expected' <<<null >/dev/null; then
  echo "model B baseline tokens changed: expected=$EXPECTED_TOKENS_B actual=$BASELINE_B" >&2
  exit 1
fi

MODEL_REGISTRY="$WORK_DIR/models.json"
cat >"$MODEL_REGISTRY" <<JSON
{
  "models": [
    {
      "id": "$MODEL_A_ID",
      "family": "llm",
      "supported_engines": ["llama.cpp"],
      "layer_count": $LAYER_COUNT_A,
      "min_memory_gb": 0,
      "preferred_compute": null,
      "placement_modes": ["pipeline_parallel"],
      "artifact_path": "$MODEL_A_PATH",
      "artifact_sha256": "$MODEL_A_SHA"
    },
    {
      "id": "$MODEL_B_ID",
      "family": "llm",
      "supported_engines": ["llama.cpp"],
      "layer_count": $LAYER_COUNT_B,
      "min_memory_gb": 0,
      "preferred_compute": null,
      "placement_modes": ["pipeline_parallel"],
      "artifact_path": "$MODEL_B_PATH",
      "artifact_sha256": "$MODEL_B_SHA"
    }
  ]
}
JSON

start_node() {
  local name=$1 node_port=$2 runtime_port=$3 peer_port=$4 role=$5
  JETSONFABRIC_CLUSTER_TOKEN="$CLUSTER_TOKEN" "$NODE_BIN" \
    --cluster-id coordinator-switch-integration \
    --node-name "$name" \
    --listen "127.0.0.1:$node_port" \
    --advertise-url "http://127.0.0.1:$node_port" \
    --data-dir "$WORK_DIR/$name" \
    --runtime-url auto \
    --runtime-bin "$RUNTIME_BIN" \
    --runtime-listen "127.0.0.1:$runtime_port" \
    --runtime-idle \
    --runtime-compute-backend cpu \
    --runtime-mode pipeline_parallel \
    --runtime-ctx-size 256 \
    --runtime-n-gpu-layers 0 \
    --runtime-threads "${JF_RUNTIME_THREADS:-2}" \
    --runtime-revision "$RUNTIME_REVISION" \
    --runtime-llama-cpp-revision "$LLAMA_CPP_COMMIT" \
    --engine llama.cpp \
    --model "$MODEL_A_ID" \
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

start_node switch-stage0 "$NODE0_PORT" "$RUNTIME0_PORT" "$NODE1_PORT" coordinator
start_node switch-stage1 "$NODE1_PORT" "$RUNTIME1_PORT" "$NODE0_PORT" worker
NODE0_URL="http://127.0.0.1:$NODE0_PORT"
NODE1_URL="http://127.0.0.1:$NODE1_PORT"

wait_for_url() {
  local url=$1
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.25
  done
  echo "timed out waiting for $url" >&2
  return 1
}
wait_for_members() {
  for _ in $(seq 1 240); do
    if [[ "$(curl -fsS "$NODE0_URL/v1/cluster/members" 2>/dev/null | jq '.members | length' 2>/dev/null || echo 0)" == "2" ]]; then return 0; fi
    sleep 0.25
  done
  echo "timed out waiting for two members" >&2
  return 1
}
wait_for_url "$NODE0_URL/healthz"
wait_for_url "$NODE1_URL/healthz"
wait_for_url "$NODE0_URL/v1/runtime/deployment"
wait_for_url "$NODE1_URL/v1/runtime/deployment"
wait_for_members
COORDINATOR_NODE_ID="$(curl -fsS "$NODE0_URL/healthz" | jq -r '.node_id')"
[[ -n "$COORDINATOR_NODE_ID" && "$COORDINATOR_NODE_ID" != "null" ]] || {
  echo "coordinator node ID is unavailable" >&2
  exit 1
}

assert_runtime_deployment() {
  local node_url=$1 deployment=$2 epoch=$3 model=$4 model_sha256=$5
  curl -fsS "$node_url/v1/runtime/deployment" | jq -e \
    --arg deployment "$deployment" --argjson epoch "$epoch" --arg model "$model" --arg model_sha256 "$model_sha256" \
    --argjson layer_count "$LAYER_COUNT_A" \
    '.resident == true and .active == true and .state == "active" and
     .deployment.deployment_id == $deployment and .deployment.epoch == $epoch and
     .deployment.model_id == $model and .deployment.model_sha256 == $model_sha256 and
     .model_memory.layer_start >= 0 and .model_memory.layer_end > .model_memory.layer_start and
     .model_memory.layer_end <= $layer_count and
     .model_memory.layer_count == $layer_count and
     .model_memory.resident_weight_bytes > 0 and
     .model_memory.resident_weight_bytes < .model_memory.total_weight_bytes and
     .model_memory.resident_tensor_count > 0 and
     .model_memory.partitioned == true and .model_memory.pinned == true' >/dev/null
}

switch_model() {
  local deployment=$1 model=$2 expected_epoch=$3 output=$4
  local code
  code="$(curl -sS -o "$output" -w '%{http_code}' \
    -X POST "$NODE0_URL/v1/deployments/switch" \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -nc --arg deployment "$deployment" --arg model "$model" '{deployment_id:$deployment,model:$model,allow_colocated_stages:true,ctx_size:256,threads:2,n_gpu_layers:0}')")"
  [[ "$code" == "200" ]] || { echo "switch to $model returned HTTP $code" >&2; cat "$output" >&2; exit 1; }
  jq -e --arg deployment "$deployment" --arg model "$model" --argjson epoch "$expected_epoch" \
    '.phase == "active" and .active.deployment_id == $deployment and .active.epoch == $epoch and .active.model.model_id == $model and (.active.stages | length) == 2' "$output" >/dev/null
}

run_model() {
  local model=$1 deployment=$2 epoch=$3 baseline=$4 output=$5 stage_count=${6:-2}
  local code
  code="$(curl -sS -o "$output" -w '%{http_code}' \
    -X POST "$NODE0_URL/v1/layer-split/run" \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -nc --arg model "$model" --arg payload "$RAW_PROMPT" --argjson max_tokens "$MAX_TOKENS" '{model:$model,payload:$payload,max_tokens:$max_tokens}')")"
  [[ "$code" == "200" ]] || { echo "inference for $model returned HTTP $code" >&2; cat "$output" >&2; exit 1; }
  jq -e --arg deployment "$deployment" --arg model "$model" --argjson epoch "$epoch" --argjson baseline "$baseline" --argjson max_tokens "$MAX_TOKENS" --argjson stage_count "$stage_count" \
    '.runtime_identity.deployment_id == $deployment and .runtime_identity.epoch == $epoch and .runtime_identity.model_id == $model and .result.sampled_tokens == $baseline and .plan.stage_count == $stage_count and (.result.stages | length) == ($stage_count * $max_tokens)' "$output" >/dev/null
}

run_runtime_owned_chat() {
  local model=$1 deployment=$2 epoch=$3 output=$4 headers=$5
  local code stage_calls remote_stage_calls
  code="$(curl -sS -D "$headers" -o "$output" -w '%{http_code}' \
    -X POST "$NODE1_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -nc --arg model "$model" '{model:$model,messages:[{role:"user",content:"hello"}],max_tokens:2}')")"
  [[ "$code" == "200" ]] || { echo "runtime-owned chat for $model returned HTTP $code" >&2; cat "$output" >&2; exit 1; }
  jq -e --arg model "$model" \
    '.object == "chat.completion" and .model == $model and (.choices | length) == 1 and .choices[0].message.role == "assistant"' \
    "$output" >/dev/null
  grep -qi "^X-JetsonFabric-Deployment-ID: $deployment" "$headers"
  grep -qi "^X-JetsonFabric-Deployment-Epoch: $epoch" "$headers"
  grep -qi '^X-JetsonFabric-Generation-Owner: runtime' "$headers"
  stage_calls="$(awk -F': ' 'tolower($1) == "x-jetsonfabric-stage-calls" {gsub("\r", "", $2); print $2}' "$headers")"
  remote_stage_calls="$(awk -F': ' 'tolower($1) == "x-jetsonfabric-remote-stage-calls" {gsub("\r", "", $2); print $2}' "$headers")"
  if [[ ! "$stage_calls" =~ ^[0-9]+$ || ! "$remote_stage_calls" =~ ^[0-9]+$ ||
        "$remote_stage_calls" -le 0 || "$stage_calls" -ne $((remote_stage_calls * 2)) ]]; then
    echo "invalid managed generation stage calls: stage_calls=$stage_calls remote_stage_calls=$remote_stage_calls" >&2
    exit 1
  fi
}

assert_stale_generation_identity_rejected() {
  local switch_output=$1 deployment=$2 stale_epoch=$3 model=$4 model_sha256=$5 output=$6
  local payload code
  payload="$(jq -c \
    --arg deployment "$deployment" \
    --argjson epoch "$stale_epoch" \
    --arg model "$model" \
    --arg model_sha256 "$model_sha256" \
    '{request_id:"stale-generation",session_id:"stale-generation",model_id:$model,prompt:"hello",max_tokens:1,deployment:{deployment_id:$deployment,epoch:$epoch,model_id:$model,model_sha256:$model_sha256},stages:.active.stages}' \
    "$switch_output")"
  code="$(curl -sS -o "$output" -w '%{http_code}' \
    -X POST "$NODE0_URL/v1/runtime/generate" \
    -H 'Content-Type: application/json' \
    -H "X-JetsonFabric-Coordinator-Node-ID: $COORDINATOR_NODE_ID" \
    -H "X-JetsonFabric-Cluster-Token: $CLUSTER_TOKEN" \
    --data-binary "$payload")"
  [[ "$code" == "200" ]] || { echo "stale generation identity returned HTTP $code" >&2; cat "$output" >&2; exit 1; }
  jq -e '.type == "error" and .code == "deployment_mismatch"' "$output" >/dev/null
}

assert_model_not_active() {
  local model=$1 output=$2
  local code
  code="$(curl -sS -o "$output" -w '%{http_code}' \
    -X POST "$NODE0_URL/v1/layer-split/run" \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -nc --arg model "$model" '{model:$model,payload:"hello",max_tokens:1}')")"
  [[ "$code" == "409" ]] || { echo "inactive model $model returned HTTP $code" >&2; cat "$output" >&2; exit 1; }
  jq -e '.error == "model_not_active"' "$output" >/dev/null
}

SWITCH_A="$WORK_DIR/switch-a.json"
RUN_A="$WORK_DIR/run-a.json"
SWITCH_B="$WORK_DIR/switch-b.json"
RUN_B="$WORK_DIR/run-b.json"
switch_model "$DEPLOYMENT_A" "$MODEL_A_ID" 1 "$SWITCH_A"
assert_runtime_deployment "$NODE0_URL" "$DEPLOYMENT_A" 1 "$MODEL_A_ID" "$MODEL_A_SHA"
assert_runtime_deployment "$NODE1_URL" "$DEPLOYMENT_A" 1 "$MODEL_A_ID" "$MODEL_A_SHA"
run_model "$MODEL_A_ID" "$DEPLOYMENT_A" 1 "$BASELINE_A" "$RUN_A"
assert_model_not_active "$MODEL_B_ID" "$WORK_DIR/model-b-before.json"

switch_model "$DEPLOYMENT_B" "$MODEL_B_ID" 2 "$SWITCH_B"
assert_runtime_deployment "$NODE0_URL" "$DEPLOYMENT_B" 2 "$MODEL_B_ID" "$MODEL_B_SHA"
assert_runtime_deployment "$NODE1_URL" "$DEPLOYMENT_B" 2 "$MODEL_B_ID" "$MODEL_B_SHA"
run_model "$MODEL_B_ID" "$DEPLOYMENT_B" 2 "$BASELINE_B" "$RUN_B"
run_runtime_owned_chat "$MODEL_B_ID" "$DEPLOYMENT_B" 2 "$WORK_DIR/chat-b.json" "$WORK_DIR/chat-b.headers"
assert_stale_generation_identity_rejected "$SWITCH_B" "$DEPLOYMENT_B" 1 "$MODEL_B_ID" "$MODEL_B_SHA" "$WORK_DIR/stale-generation.jsonl"
assert_model_not_active "$MODEL_A_ID" "$WORK_DIR/model-a-after.json"

curl -fsS "$NODE0_URL/v1/deployments/active" | jq -e \
  --arg deployment "$DEPLOYMENT_B" --arg model "$MODEL_B_ID" \
  '.phase == "active" and .active.deployment_id == $deployment and .active.epoch == 2 and .active.model.model_id == $model and .in_flight == 0' >/dev/null

kill "${PIDS[1]}" 2>/dev/null || true
wait "${PIDS[1]}" 2>/dev/null || true
RECONCILED_STATUS="$WORK_DIR/reconciled-single.json"
for _ in $(seq 1 240); do
  if curl -fsS "$NODE0_URL/v1/deployments/active" >"$RECONCILED_STATUS" 2>/dev/null &&
     jq -e --arg model "$MODEL_B_ID" \
       '.active.epoch == 3 and .active.model.model_id == $model and (.active.stages | length) == 1 and (.phase == "active" or .phase == "draining")' \
       "$RECONCILED_STATUS" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
jq -e --arg model "$MODEL_B_ID" \
  '.active.epoch == 3 and .active.model.model_id == $model and (.active.stages | length) == 1 and (.phase == "active" or .phase == "draining")' \
  "$RECONCILED_STATUS" >/dev/null
RECONCILED_DEPLOYMENT="$(jq -r '.active.deployment_id' "$RECONCILED_STATUS")"
run_model "$MODEL_B_ID" "$RECONCILED_DEPLOYMENT" 3 "$BASELINE_B" "$WORK_DIR/run-b-reconciled.json" 1

echo "Coordinator model switch validation passed"
echo "Deployment A: $DEPLOYMENT_A model=$MODEL_A_ID sha=$MODEL_A_SHA"
echo "Deployment B: $DEPLOYMENT_B model=$MODEL_B_ID sha=$MODEL_B_SHA"
echo "Lifecycle: deploy A -> infer A -> prepare/activate B -> drain/unload A -> infer B -> lose worker -> auto-reconcile B to one stage"
