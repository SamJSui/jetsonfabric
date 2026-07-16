#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
BIN_DIR="$WORK_DIR/bin"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$BIN_DIR" "$LOG_DIR"

MODEL_PATH="${CI_MODEL_PATH:?CI_MODEL_PATH is required}"
LLAMA_CPP_DIR="${CI_LLAMA_CPP_DIR:-$ROOT_DIR/runtime/third_party/llama.cpp}"
LLAMA_CPP_COMMIT="${CI_LLAMA_CPP_COMMIT:?CI_LLAMA_CPP_COMMIT is required}"
RUNTIME_BUILD_DIR="${CI_RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-ci-cpu}"
NODE0_PORT="${CI_NODE0_PORT:-18180}"
NODE1_PORT="${CI_NODE1_PORT:-18181}"

PIDS=()
cleanup() {
  local status=$?
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "Two-node real CPU E2E test failed. Logs:" >&2
    for log_file in "$LOG_DIR"/*.log; do
      [[ -e "$log_file" ]] || continue
      echo "===== $log_file =====" >&2
      cat "$log_file" >&2
    done
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

cmake -S runtime -B "$RUNTIME_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DJF_LLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_DIR" \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF
cmake --build "$RUNTIME_BUILD_DIR" --parallel 2

go build -buildvcs=false -o "$BIN_DIR/jetsonfabric-node" ./cmd/jetsonfabric-node

RUNTIME_BIN="$RUNTIME_BUILD_DIR/jetsonfabric-runtime-worker"
if [[ ! -x "$RUNTIME_BIN" ]]; then
  echo "Runtime binary missing after build: $RUNTIME_BIN" >&2
  exit 1
fi

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
  --model qwen2.5-coder-1.5b-q4 \
  --model-path "$MODEL_PATH" \
  --stage-index 0 \
  --stage-count 2 \
  --layer-start 0 \
  --layer-end 14 \
  --role coordinator \
  --discovery static \
  --seeds "http://127.0.0.1:$NODE1_PORT" \
  --discovery-interval 250ms \
  --stale-after 5s \
  --benchmarks "$WORK_DIR/benchmarks-stage0.jsonl" \
  --models configs/models.example.json \
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
  --model qwen2.5-coder-1.5b-q4 \
  --model-path "$MODEL_PATH" \
  --stage-index 1 \
  --stage-count 2 \
  --layer-start 14 \
  --layer-end 28 \
  --role worker \
  --discovery static \
  --seeds "http://127.0.0.1:$NODE0_PORT" \
  --discovery-interval 250ms \
  --stale-after 5s \
  --benchmarks "$WORK_DIR/benchmarks-stage1.jsonl" \
  --models configs/models.example.json \
  >"$LOG_DIR/node-stage1.log" 2>&1 &
PIDS+=("$!")

wait_for_url "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"
wait_for_url "http://127.0.0.1:$NODE1_PORT/v1/cluster/members"
wait_for_two_members "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"

members_json="$(curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/cluster/members")"
jq -e '
  (.members | length) == 2 and
  (any(.members[]; .node_name == "ci-stage0" and .capabilities.runtime_stage_index == 0 and .capabilities.runtime_stage_count == 2 and .capabilities.runtime_layer_start == 0 and .capabilities.runtime_layer_end == 14)) and
  (any(.members[]; .node_name == "ci-stage1" and .capabilities.runtime_stage_index == 1 and .capabilities.runtime_stage_count == 2 and .capabilities.runtime_layer_start == 14 and .capabilities.runtime_layer_end == 28))
' <<<"$members_json" >/dev/null

preview_json="$(curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/routes/preview?model=qwen2.5-coder-1.5b-q4&stage_count=2&allow_colocated_stages=true")"
jq -e '
  .valid == true and
  .mode == "pipeline_parallel" and
  .topology == "colocated" and
  .stage_count == 2 and
  .logical_node_count == 2 and
  .physical_host_count == 1 and
  (.stages | length) == 2 and
  .stages[0].stage_index == 0 and
  .stages[0].stage_count == 2 and
  .stages[0].node_name == "ci-stage0" and
  .stages[0].layer_start == 0 and
  .stages[0].layer_end == 14 and
  .stages[1].stage_index == 1 and
  .stages[1].stage_count == 2 and
  .stages[1].node_name == "ci-stage1" and
  .stages[1].layer_start == 14 and
  .stages[1].layer_end == 28
' <<<"$preview_json" >/dev/null

response_file="$WORK_DIR/response.json"
http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$NODE0_PORT/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "ci-real-runtime-e2e",
    "model": "qwen2.5-coder-1.5b-q4",
    "payload": "Once upon a time",
    "max_tokens": 4,
    "stage_count": 2,
    "allow_colocated_stages": true
  }')"

if [[ "$http_code" != "200" ]]; then
  echo "Expected HTTP 200, got $http_code" >&2
  cat "$response_file" >&2
  exit 1
fi

jq -e '
  .inter_stage_payload_kind == "text" and
  .plan.mode == "pipeline_parallel" and
  .plan.topology == "colocated" and
  .plan.stage_count == 2 and
  (.plan.stages | length) == 2 and
  .result.payload_kind == "text" and
  (.result.stages | length) == 2 and
  .result.stages[0].status_code == 200 and
  .result.stages[1].status_code == 200 and
  .result.stages[0].payload_kind_in == "text" and
  .result.stages[0].payload_kind_out == "text" and
  .result.stages[0].payload_out > 0 and
  .result.stages[1].payload_in == .result.stages[0].payload_out and
  (.result.payload | type == "string") and
  (.result.payload | length > 0)
' "$response_file" >/dev/null

echo "Two-node real CPU E2E test passed"
jq '{inter_stage_payload_kind, plan: {mode: .plan.mode, topology: .plan.topology, stage_count: .plan.stage_count, physical_host_count: .plan.physical_host_count, stages: .plan.stages}, result: {payload_kind: .result.payload_kind, payload: .result.payload, prompt_tokens: .result.prompt_tokens, completion_tokens: .result.completion_tokens, stages: .result.stages}}' "$response_file"
