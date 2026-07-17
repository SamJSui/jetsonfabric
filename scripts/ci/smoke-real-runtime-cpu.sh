#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
BIN_DIR="$WORK_DIR/bin"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_LOG_DIR="${CI_ARTIFACT_LOG_DIR:-$ROOT_DIR/.ci-logs/real-runtime}"
mkdir -p "$BIN_DIR" "$LOG_DIR"

MODEL_PATH="${CI_MODEL_PATH:?CI_MODEL_PATH is required}"
LLAMA_CPP_DIR="${CI_LLAMA_CPP_DIR:-$ROOT_DIR/runtime/third_party/llama.cpp}"
LLAMA_CPP_COMMIT="${CI_LLAMA_CPP_COMMIT:?CI_LLAMA_CPP_COMMIT is required}"
RUNTIME_BUILD_DIR="${CI_RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-ci-cpu}"
NODE0_PORT="${CI_NODE0_PORT:-18180}"
NODE1_PORT="${CI_NODE1_PORT:-18181}"
MODEL_ID="ci-tiny-llama"
response_file=""

PIDS=()
cleanup() {
  local status=$?
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "Two-node real partial-layer CPU E2E failed. Logs:" >&2
    for log_file in "$LOG_DIR"/*.log; do
      [[ -e "$log_file" ]] || continue
      echo "===== $log_file =====" >&2
      cat "$log_file" >&2
    done
    rm -rf "$ARTIFACT_LOG_DIR"
    mkdir -p "$ARTIFACT_LOG_DIR"
    cp -a "$LOG_DIR"/. "$ARTIFACT_LOG_DIR"/ 2>/dev/null || true
    if [[ -n "$response_file" && -f "$response_file" ]]; then
      cp "$response_file" "$ARTIFACT_LOG_DIR/response.json"
    fi
  fi
  rm -rf "$WORK_DIR"
  exit "$status"
}
trap cleanup EXIT

wait_for_url() {
  local url=$1
  local attempts=${2:-240}
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

wait_for_two_members() {
  local url=$1
  local attempts=${2:-240}
  for ((i = 1; i <= attempts; i++)); do
    local count
    count="$(curl -fsS "$url" 2>/dev/null | jq '.members | length' 2>/dev/null || echo 0)"
    if [[ "$count" == "2" ]]; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for two members at $url" >&2
  return 1
}

cd "$ROOT_DIR"
rm -rf "$ARTIFACT_LOG_DIR"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "CI model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ "$(head -c 4 "$MODEL_PATH")" != "GGUF" ]]; then
  echo "CI model is not a GGUF file: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  rm -rf "$LLAMA_CPP_DIR"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_COMMIT"
git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_COMMIT"
git -C "$LLAMA_CPP_DIR" reset --hard "$LLAMA_CPP_COMMIT"

cmake -S runtime -B "$RUNTIME_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DJF_LLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_DIR" \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF \
  2>&1 | tee "$LOG_DIR/cmake-configure.log"
cmake --build "$RUNTIME_BUILD_DIR" --parallel 2 \
  2>&1 | tee "$LOG_DIR/cmake-build.log"

go build -buildvcs=false -o "$BIN_DIR/jetsonfabric-node" ./cmd/jetsonfabric-node

RUNTIME_BIN="$RUNTIME_BUILD_DIR/jetsonfabric-runtime-worker"
STAGE_TEST_BIN="$RUNTIME_BUILD_DIR/jetsonfabric-llama-stage-test"
if [[ ! -x "$RUNTIME_BIN" || ! -x "$STAGE_TEST_BIN" ]]; then
  echo "Runtime or stage-test binary missing after build" >&2
  exit 1
fi

if ! CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" >"$LOG_DIR/llama-stage-equivalence.log" 2>&1; then
  cat "$LOG_DIR/llama-stage-equivalence.log" >&2
  exit 1
fi
LAYER_COUNT="$(CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" --print-layer-count)"
BASELINE_TOKENS="$(CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" --baseline-tokens)"
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 2 ]]; then
  echo "Invalid model layer count: $LAYER_COUNT" >&2
  exit 1
fi
if ! jq -e 'type == "array" and length == 2 and all(.[]; type == "number")' <<<"$BASELINE_TOKENS" >/dev/null; then
  echo "Invalid two-token baseline: $BASELINE_TOKENS" >&2
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

"$BIN_DIR/jetsonfabric-node" \
  --cluster-id ci-real-runtime-e2e \
  --node-name ci-stage0 \
  --listen "127.0.0.1:$NODE0_PORT" \
  --advertise-url "http://127.0.0.1:$NODE0_PORT" \
  --data-dir "$WORK_DIR/node-stage0" \
  --runtime-url auto \
  --runtime-bin "$RUNTIME_BIN" \
  --runtime-listen 127.0.0.1:0 \
  --runtime-compute-backend cpu \
  --runtime-mode pipeline_parallel \
  --runtime-ctx-size 256 \
  --runtime-n-gpu-layers 0 \
  --runtime-threads 2 \
  --engine llama.cpp \
  --model "$MODEL_ID" \
  --model-path "$MODEL_PATH" \
  --stage-index 0 \
  --stage-count 2 \
  --layer-start 0 \
  --layer-end "$SPLIT_LAYER" \
  --role coordinator \
  --discovery static \
  --seeds "http://127.0.0.1:$NODE1_PORT" \
  --discovery-interval 250ms \
  --stale-after 5s \
  --benchmarks "$WORK_DIR/benchmarks-stage0.jsonl" \
  --models "$MODEL_REGISTRY" \
  >"$LOG_DIR/node-stage0.log" 2>&1 &
PIDS+=("$!")

"$BIN_DIR/jetsonfabric-node" \
  --cluster-id ci-real-runtime-e2e \
  --node-name ci-stage1 \
  --listen "127.0.0.1:$NODE1_PORT" \
  --advertise-url "http://127.0.0.1:$NODE1_PORT" \
  --data-dir "$WORK_DIR/node-stage1" \
  --runtime-url auto \
  --runtime-bin "$RUNTIME_BIN" \
  --runtime-listen 127.0.0.1:0 \
  --runtime-compute-backend cpu \
  --runtime-mode pipeline_parallel \
  --runtime-ctx-size 256 \
  --runtime-n-gpu-layers 0 \
  --runtime-threads 2 \
  --engine llama.cpp \
  --model "$MODEL_ID" \
  --model-path "$MODEL_PATH" \
  --stage-index 1 \
  --stage-count 2 \
  --layer-start "$SPLIT_LAYER" \
  --layer-end "$LAYER_COUNT" \
  --role worker \
  --discovery static \
  --seeds "http://127.0.0.1:$NODE0_PORT" \
  --discovery-interval 250ms \
  --stale-after 5s \
  --benchmarks "$WORK_DIR/benchmarks-stage1.jsonl" \
  --models "$MODEL_REGISTRY" \
  >"$LOG_DIR/node-stage1.log" 2>&1 &
PIDS+=("$!")

wait_for_url "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"
wait_for_url "http://127.0.0.1:$NODE1_PORT/v1/cluster/members"
wait_for_two_members "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"

members_json="$(curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/cluster/members")"
jq -e --argjson split "$SPLIT_LAYER" --argjson layers "$LAYER_COUNT" '
  (.members | length) == 2 and
  (any(.members[]; .node_name == "ci-stage0" and .capabilities.runtime_stage_index == 0 and .capabilities.runtime_stage_count == 2 and .capabilities.runtime_layer_start == 0 and .capabilities.runtime_layer_end == $split)) and
  (any(.members[]; .node_name == "ci-stage1" and .capabilities.runtime_stage_index == 1 and .capabilities.runtime_stage_count == 2 and .capabilities.runtime_layer_start == $split and .capabilities.runtime_layer_end == $layers))
' <<<"$members_json" >/dev/null

preview_json="$(curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/routes/preview?model=$MODEL_ID&stage_count=2&allow_colocated_stages=true")"
jq -e --argjson split "$SPLIT_LAYER" --argjson layers "$LAYER_COUNT" '
  .valid == true and
  .mode == "pipeline_parallel" and
  .stage_count == 2 and
  (.stages | length) == 2 and
  .stages[0].layer_start == 0 and
  .stages[0].layer_end == $split and
  .stages[1].layer_start == $split and
  .stages[1].layer_end == $layers
' <<<"$preview_json" >/dev/null

response_file="$WORK_DIR/response.json"
http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$NODE0_PORT/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  -d "{
    \"request_id\": \"ci-real-runtime-e2e\",
    \"model\": \"$MODEL_ID\",
    \"payload\": \"Once upon a time\",
    \"max_tokens\": 2,
    \"stage_count\": 2,
    \"allow_colocated_stages\": true,
    \"strict_payload_transitions\": true
  }")"

if [[ "$http_code" != "200" ]]; then
  echo "Expected HTTP 200, got $http_code" >&2
  cat "$response_file" >&2
  exit 1
fi

jq -e --argjson baseline "$BASELINE_TOKENS" '
  .inter_stage_payload_kind == "activation" and
  .plan.mode == "pipeline_parallel" and
  .plan.stage_count == 2 and
  .result.payload_kind == "sampled_token" and
  .result.sampled_tokens == $baseline and
  .result.sampled_token == $baseline[1] and
  (.result.stages | length) == 4 and
  .result.stages[0].status_code == 200 and
  .result.stages[1].status_code == 200 and
  .result.stages[2].status_code == 200 and
  .result.stages[3].status_code == 200 and
  .result.stages[0].phase == "prefill" and
  .result.stages[0].decode_step == 0 and
  .result.stages[0].payload_kind_in == "text" and
  .result.stages[0].payload_kind_out == "activation" and
  .result.stages[1].phase == "prefill" and
  .result.stages[1].decode_step == 0 and
  .result.stages[1].payload_kind_in == "activation" and
  .result.stages[1].payload_kind_out == "sampled_token" and
  .result.stages[2].phase == "decode" and
  .result.stages[2].decode_step == 1 and
  .result.stages[2].payload_kind_in == "sampled_token" and
  .result.stages[2].payload_kind_out == "activation" and
  .result.stages[3].phase == "decode" and
  .result.stages[3].decode_step == 1 and
  .result.stages[3].payload_kind_in == "activation" and
  .result.stages[3].payload_kind_out == "sampled_token" and
  .result.stages[0].payload_out == .result.stages[1].payload_in and
  .result.stages[0].payload_crc32_out == .result.stages[1].payload_crc32_in and
  .result.stages[2].payload_out == .result.stages[3].payload_in and
  .result.stages[2].payload_crc32_out == .result.stages[3].payload_crc32_in
' "$response_file" >/dev/null

echo "Two-node real partial-layer prefill/decode CPU E2E passed"
jq '{inter_stage_payload_kind, plan, result}' "$response_file"
