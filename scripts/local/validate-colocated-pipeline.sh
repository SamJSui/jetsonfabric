#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GGUF to validate}"
MODEL_ID="${MODEL_ID:-qwen2.5-coder-1.5b-q4}"
RAW_PROMPT="${JF_RAW_PROMPT:-Once upon a time}"
CHAT_PROMPT="${JF_CHAT_PROMPT:-Explain JetsonFabric in one sentence.}"
MAX_TOKENS="${JF_MAX_TOKENS:-2}"
NODE0_PORT="${JF_NODE0_PORT:-19180}"
NODE1_PORT="${JF_NODE1_PORT:-19181}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build}"
RUNTIME_BIN="${RUNTIME_BIN:-$ROOT_DIR/dist/jetsonfabric-runtime-worker}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/dist/jetsonfabric-node}"
REPORT_PATH="${JF_REPORT_PATH:-$ROOT_DIR/reports/phase1-colocated.json}"
WORK_DIR="$(mktemp -d)"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR" "$(dirname "$REPORT_PATH")"

PIDS=()
cleanup() {
  local status=$?
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "Phase 1 colocated validation failed. Logs are in $LOG_DIR until this script exits." >&2
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

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}
for command in curl jq sha256sum awk go cmake; do
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
BASELINE_TOKENS="$(CI_MODEL_PATH="$MODEL_PATH" "$STAGE_TEST_BIN" --baseline-tokens)"
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 2 ]]; then
  echo "invalid layer count: $LAYER_COUNT" >&2
  exit 1
fi
if ! jq -e 'type == "array" and length == 2 and all(.[]; type == "number")' <<<"$BASELINE_TOKENS" >/dev/null; then
  echo "invalid baseline tokens: $BASELINE_TOKENS" >&2
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
  local name=$1 port=$2 peer_port=$3 stage_index=$4 layer_start=$5 layer_end=$6 role=$7
  "$NODE_BIN" \
    --cluster-id phase1-colocated \
    --node-name "$name" \
    --listen "127.0.0.1:$port" \
    --advertise-url "http://127.0.0.1:$port" \
    --data-dir "$WORK_DIR/$name" \
    --runtime-url auto \
    --runtime-bin "$RUNTIME_BIN" \
    --runtime-listen 127.0.0.1:0 \
    --runtime-compute-backend cpu \
    --runtime-mode pipeline_parallel \
    --runtime-ctx-size "${JF_CTX_SIZE:-4096}" \
    --runtime-n-gpu-layers 0 \
    --runtime-threads "${JF_RUNTIME_THREADS:-0}" \
    --engine llama.cpp \
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

start_node phase1-stage0 "$NODE0_PORT" "$NODE1_PORT" 0 0 "$SPLIT_LAYER" coordinator
start_node phase1-stage1 "$NODE1_PORT" "$NODE0_PORT" 1 "$SPLIT_LAYER" "$LAYER_COUNT" worker

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
  echo "timed out waiting for two members" >&2
  return 1
}

wait_for_url "http://127.0.0.1:$NODE0_PORT/healthz"
wait_for_url "http://127.0.0.1:$NODE1_PORT/healthz"
wait_for_two_members "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"

MEMBERS_FILE="$WORK_DIR/members.json"
PREVIEW_FILE="$WORK_DIR/preview.json"
DIAGNOSTIC_FILE="$WORK_DIR/diagnostic.json"
CHAT_FILE="$WORK_DIR/chat.json"
CHAT_HEADERS="$WORK_DIR/chat.headers"
curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/cluster/members" >"$MEMBERS_FILE"

jq -e --arg model "$MODEL_ID" --arg sha "$MODEL_SHA256" '
  (.members | length) == 2 and
  all(.members[];
    .capabilities.runtime_model_id == $model and
    .capabilities.runtime_model_sha256 == $sha and
    .capabilities.runtime_engine == "llama.cpp" and
    .capabilities.runtime_compute_backend == "cpu" and
    .capabilities.runtime_execution_mode == "pipeline_parallel")
' "$MEMBERS_FILE" >/dev/null

curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/routes/preview?model=$MODEL_ID&stage_count=2&allow_colocated_stages=true" >"$PREVIEW_FILE"
jq -e '
  .valid == true and .mode == "pipeline_parallel" and
  .topology == "colocated" and .stage_count == 2 and
  .logical_node_count == 2 and .physical_host_count == 1
' "$PREVIEW_FILE" >/dev/null

curl -fsS -X POST "http://127.0.0.1:$NODE0_PORT/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg payload "$RAW_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{request_id:"phase1-diagnostic",model:$model,payload:$payload,max_tokens:$max_tokens,stage_count:2,allow_colocated_stages:true}')" \
  >"$DIAGNOSTIC_FILE"

jq -e --arg sha "$MODEL_SHA256" --argjson baseline "$BASELINE_TOKENS" '
  .runtime_identity.model_sha256 == $sha and
  .inter_stage_payload_kind == "activation" and
  .result.sampled_tokens == $baseline and
  (.result.stages | length) == 4 and
  .result.stages[0].payload_crc32_out == .result.stages[1].payload_crc32_in and
  .result.stages[2].payload_crc32_out == .result.stages[3].payload_crc32_in
' "$DIAGNOSTIC_FILE" >/dev/null

CHAT_SECONDS="$(curl -fsS -D "$CHAT_HEADERS" -o "$CHAT_FILE" -w '%{time_total}' \
  -X POST "http://127.0.0.1:$NODE1_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg prompt "$CHAT_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,jetsonfabric:{stage_count:2,allow_colocated_stages:true}}')")"
CHAT_DURATION_MS="$(awk -v seconds="$CHAT_SECONDS" 'BEGIN { printf "%.3f", seconds * 1000 }')"

jq -e --arg model "$MODEL_ID" '
  .object == "chat.completion" and .model == $model and
  (.choices | length) == 1 and .choices[0].message.role == "assistant" and
  (.choices[0].finish_reason == "stop" or .choices[0].finish_reason == "length")
' "$CHAT_FILE" >/dev/null
grep -qi "^X-JetsonFabric-Model-SHA256: $MODEL_SHA256" "$CHAT_HEADERS"
grep -qi '^X-JetsonFabric-Topology: colocated' "$CHAT_HEADERS"

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -n \
  --arg timestamp "$TIMESTAMP" \
  --arg model_id "$MODEL_ID" \
  --arg model_path "$MODEL_PATH" \
  --arg model_sha256 "$MODEL_SHA256" \
  --arg raw_prompt "$RAW_PROMPT" \
  --arg chat_prompt "$CHAT_PROMPT" \
  --arg chat_duration_ms "$CHAT_DURATION_MS" \
  --argjson layer_count "$LAYER_COUNT" \
  --argjson split_layer "$SPLIT_LAYER" \
  --argjson baseline_tokens "$BASELINE_TOKENS" \
  --slurpfile members "$MEMBERS_FILE" \
  --slurpfile preview "$PREVIEW_FILE" \
  --slurpfile diagnostic "$DIAGNOSTIC_FILE" \
  --slurpfile chat "$CHAT_FILE" \
  '{
    schema: "jetsonfabric.phase1.colocated.v1",
    timestamp: $timestamp,
    model: {
      id: $model_id,
      path: $model_path,
      sha256: $model_sha256,
      layer_count: $layer_count,
      split_layer: $split_layer
    },
    topology: $preview[0],
    membership: $members[0].members,
    correctness: {
      raw_prompt: $raw_prompt,
      baseline_tokens: $baseline_tokens,
      distributed_tokens: $diagnostic[0].result.sampled_tokens,
      tokens_match: ($diagnostic[0].result.sampled_tokens == $baseline_tokens),
      activation_crc_continuity: (
        $diagnostic[0].result.stages[0].payload_crc32_out == $diagnostic[0].result.stages[1].payload_crc32_in and
        $diagnostic[0].result.stages[2].payload_crc32_out == $diagnostic[0].result.stages[3].payload_crc32_in
      )
    },
    performance: {
      stage_engine_latency_ms_total: ([$diagnostic[0].result.stages[].latency_ms] | add // 0),
      activation_bytes_prefill: $diagnostic[0].result.stages[0].payload_out,
      activation_bytes_decode: $diagnostic[0].result.stages[2].payload_out,
      openai_request_duration_ms: ($chat_duration_ms | tonumber),
      stage_traces: $diagnostic[0].result.stages
    },
    openai: {
      prompt: $chat_prompt,
      response: $chat[0]
    }
  }' >"$REPORT_PATH"

echo "Phase 1 colocated validation passed"
echo "Model SHA-256: $MODEL_SHA256"
echo "Report: $REPORT_PATH"
jq '{model, topology: {topology: .topology.topology, stage_count: .topology.stage_count}, correctness, performance: {stage_engine_latency_ms_total, activation_bytes_prefill, activation_bytes_decode, openai_request_duration_ms}, openai: .openai.response}' "$REPORT_PATH"
