#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/process-memory.sh"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GGUF to validate}"
MODEL_ID="${MODEL_ID:-qwen2.5-coder-1.5b-q4}"
RAW_PROMPT="${JF_RAW_PROMPT:-Once upon a time}"
CHAT_PROMPT="${JF_CHAT_PROMPT:-Explain JetsonFabric in one sentence.}"
MAX_TOKENS="${JF_MAX_TOKENS:-4}"
EXPECTED_TOKENS="${JF_EXPECTED_TOKENS:-}"
EXPECTED_FINISH_REASON="${JF_EXPECTED_FINISH_REASON:-}"
EXPECTED_CHAT_FINISH_REASON="${JF_EXPECTED_CHAT_FINISH_REASON:-}"
EXPECTED_CHAT_COMPLETION_TOKENS="${JF_EXPECTED_CHAT_COMPLETION_TOKENS:-}"
NODE0_PORT="${JF_NODE0_PORT:-19180}"
NODE1_PORT="${JF_NODE1_PORT:-19181}"
RUNTIME0_PORT="${JF_RUNTIME0_PORT:-${JF_RUNTIME_PORT:-19190}}"
RUNTIME1_PORT="${JF_RUNTIME1_PORT:-19191}"
RUNTIME_COMPUTE_BACKEND="${JF_RUNTIME_COMPUTE_BACKEND:-cpu}"
RUNTIME_N_GPU_LAYERS="${JF_RUNTIME_N_GPU_LAYERS:-0}"
RUNTIME_CUDA_ACTIVE="${JF_RUNTIME_CUDA_ACTIVE:-false}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build}"
RUNTIME_BIN="${RUNTIME_BIN:-$ROOT_DIR/dist/jetsonfabric-runtime-worker}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/dist/jetsonfabric-node}"
REPORT_PATH="${JF_REPORT_PATH:-$ROOT_DIR/reports/phase1-colocated.json}"
CLUSTER_TOKEN="${JF_CLUSTER_TOKEN:-jetsonfabric-integration-token}"

case "$RUNTIME_CUDA_ACTIVE" in
  true|false) ;;
  *)
    echo "JF_RUNTIME_CUDA_ACTIVE must be true or false" >&2
    exit 2
    ;;
esac
case "$EXPECTED_FINISH_REASON" in
  ""|length|stop) ;;
  *)
    echo "JF_EXPECTED_FINISH_REASON must be length or stop" >&2
    exit 2
    ;;
esac
case "$EXPECTED_CHAT_FINISH_REASON" in
  ""|length|stop) ;;
  *)
    echo "JF_EXPECTED_CHAT_FINISH_REASON must be length or stop" >&2
    exit 2
    ;;
esac
if [[ -n "$EXPECTED_CHAT_COMPLETION_TOKENS" && ! "$EXPECTED_CHAT_COMPLETION_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "JF_EXPECTED_CHAT_COMPLETION_TOKENS must be a non-negative integer" >&2
  exit 2
fi

RUNTIME_CUDA_ACTIVE_ARGS=()
if [[ "$RUNTIME_CUDA_ACTIVE" == "true" ]]; then
  RUNTIME_CUDA_ACTIVE_ARGS+=(--runtime-cuda-active)
fi

WORK_DIR="$(mktemp -d)"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR" "$(dirname "$REPORT_PATH")"

PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
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
trap 'exit 130' INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}
for command in curl jq sha256sum awk sed go cmake ss grep head seq; do
  require_command "$command"
done

port_is_listening() {
  local port=$1
  ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .
}

for port_spec in \
  "stage0-node:$NODE0_PORT" \
  "stage1-node:$NODE1_PORT" \
  "stage0-runtime:$RUNTIME0_PORT" \
  "stage1-runtime:$RUNTIME1_PORT"; do
  label=${port_spec%%:*}
  port=${port_spec##*:}
  if port_is_listening "$port"; then
    echo "colocated integration $label port $port is already occupied" >&2
    ss -H -ltnp "sport = :$port" >&2 2>/dev/null || true
    exit 2
  fi
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
if [[ ! "$LAYER_COUNT" =~ ^[0-9]+$ || "$LAYER_COUNT" -lt 2 ]]; then
  echo "invalid layer count: $LAYER_COUNT" >&2
  exit 1
fi
if ! jq -e --argjson count "$MAX_TOKENS" 'type == "array" and length <= $count and all(.[]; type == "number")' <<<"$BASELINE_TOKENS" >/dev/null; then
  echo "invalid baseline tokens: $BASELINE_TOKENS" >&2
  exit 1
fi
if [[ -n "$EXPECTED_TOKENS" ]] && ! jq -e --argjson actual "$BASELINE_TOKENS" --argjson expected "$EXPECTED_TOKENS" '$actual == $expected' <<<null >/dev/null; then
  echo "baseline tokens do not match JF_EXPECTED_TOKENS: expected=$EXPECTED_TOKENS actual=$BASELINE_TOKENS" >&2
  exit 1
fi
BASELINE_TOKEN_COUNT="$(jq 'length' <<<"$BASELINE_TOKENS")"
BASELINE_FINISH_REASON=length
GENERATION_PASSES="$BASELINE_TOKEN_COUNT"
if [[ "$BASELINE_TOKEN_COUNT" -lt "$MAX_TOKENS" ]]; then
  BASELINE_FINISH_REASON=stop
  GENERATION_PASSES=$((BASELINE_TOKEN_COUNT + 1))
fi
if [[ -n "$EXPECTED_FINISH_REASON" && "$BASELINE_FINISH_REASON" != "$EXPECTED_FINISH_REASON" ]]; then
  echo "baseline finish reason does not match JF_EXPECTED_FINISH_REASON: expected=$EXPECTED_FINISH_REASON actual=$BASELINE_FINISH_REASON" >&2
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
  local name=$1 node_port=$2 runtime_port=$3 peer_port=$4 stage_index=$5 layer_start=$6 layer_end=$7 role=$8
  JETSONFABRIC_CLUSTER_TOKEN="$CLUSTER_TOKEN" "$NODE_BIN" \
    --cluster-id phase1-colocated \
    --node-name "$name" \
    --listen "127.0.0.1:$node_port" \
    --advertise-url "http://127.0.0.1:$node_port" \
    --data-dir "$WORK_DIR/$name" \
    --runtime-url auto \
    --runtime-bin "$RUNTIME_BIN" \
    --runtime-listen "127.0.0.1:$runtime_port" \
    --runtime-compute-backend "$RUNTIME_COMPUTE_BACKEND" \
    "${RUNTIME_CUDA_ACTIVE_ARGS[@]}" \
    --runtime-mode pipeline_parallel \
    --runtime-ctx-size "${JF_CTX_SIZE:-4096}" \
    --runtime-n-gpu-layers "$RUNTIME_N_GPU_LAYERS" \
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

start_node phase1-stage0 "$NODE0_PORT" "$RUNTIME0_PORT" "$NODE1_PORT" 0 0 "$SPLIT_LAYER" coordinator
start_node phase1-stage1 "$NODE1_PORT" "$RUNTIME1_PORT" "$NODE0_PORT" 1 "$SPLIT_LAYER" "$LAYER_COUNT" worker

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
wait_for_url "http://127.0.0.1:$RUNTIME0_PORT/healthz"
wait_for_url "http://127.0.0.1:$RUNTIME1_PORT/healthz"
wait_for_two_members "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"

UNAUTHENTICATED_STAGE_FILE="$WORK_DIR/unauthenticated-stage.json"
UNAUTHENTICATED_STAGE_STATUS="$(curl -sS -o "$UNAUTHENTICATED_STAGE_FILE" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$NODE1_PORT/v1/layer-split/stage" \
  -H 'Content-Type: application/vnd.jetsonfabric.stage.v1+octet-stream' \
  --data-binary '')"
if [[ "$UNAUTHENTICATED_STAGE_STATUS" != "403" ]] ||
   ! jq -e '.error == "cluster_authentication_required"' "$UNAUTHENTICATED_STAGE_FILE" >/dev/null; then
  echo "unauthenticated Stagewire request was not rejected: HTTP $UNAUTHENTICATED_STAGE_STATUS" >&2
  cat "$UNAUTHENTICATED_STAGE_FILE" >&2
  exit 1
fi

MEMBERS_FILE="$WORK_DIR/members.json"
RUNTIME0_STATUS_FILE="$WORK_DIR/runtime0-status.json"
RUNTIME1_STATUS_FILE="$WORK_DIR/runtime1-status.json"
PREVIEW_FILE="$WORK_DIR/preview.json"
DIAGNOSTIC_FILE="$WORK_DIR/diagnostic.json"
RUNTIME_GENERATION_FILE="$WORK_DIR/runtime-generation.jsonl"
CHAT_FILE="$WORK_DIR/chat.json"
CHAT_HEADERS="$WORK_DIR/chat.headers"
STREAM_FILE="$WORK_DIR/chat-stream.sse"
STREAM_EVENTS_FILE="$WORK_DIR/chat-stream-events.jsonl"
curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/cluster/members" >"$MEMBERS_FILE"
curl -fsS "http://127.0.0.1:$RUNTIME0_PORT/v1/deployment" >"$RUNTIME0_STATUS_FILE"
curl -fsS "http://127.0.0.1:$RUNTIME1_PORT/v1/deployment" >"$RUNTIME1_STATUS_FILE"

jq -e --argjson layer_end "$SPLIT_LAYER" --argjson layer_count "$LAYER_COUNT" '
  .resident == true and .active == true and .state == "active" and
  .model_memory.layer_start == 0 and
  .model_memory.layer_end == $layer_end and
  .model_memory.layer_count == $layer_count and
  .model_memory.resident_weight_bytes > 0 and
  .model_memory.resident_weight_bytes < .model_memory.total_weight_bytes and
  .model_memory.resident_tensor_count > 0 and
  .model_memory.partitioned == true and
  .model_memory.pinned == true
' "$RUNTIME0_STATUS_FILE" >/dev/null
jq -e --argjson layer_start "$SPLIT_LAYER" --argjson layer_count "$LAYER_COUNT" '
  .resident == true and .active == true and .state == "active" and
  .model_memory.layer_start == $layer_start and
  .model_memory.layer_end == $layer_count and
  .model_memory.layer_count == $layer_count and
  .model_memory.resident_weight_bytes > 0 and
  .model_memory.resident_weight_bytes < .model_memory.total_weight_bytes and
  .model_memory.resident_tensor_count > 0 and
  .model_memory.partitioned == true and
  .model_memory.pinned == true
' "$RUNTIME1_STATUS_FILE" >/dev/null

TOTAL_WEIGHT0="$(jq -r '.model_memory.total_weight_bytes' "$RUNTIME0_STATUS_FILE")"
TOTAL_WEIGHT1="$(jq -r '.model_memory.total_weight_bytes' "$RUNTIME1_STATUS_FILE")"
if [[ "$TOTAL_WEIGHT0" != "$TOTAL_WEIGHT1" ]]; then
  echo "stages disagree on full model weight size: stage0=$TOTAL_WEIGHT0 stage1=$TOTAL_WEIGHT1" >&2
  exit 1
fi

jq -e \
  --arg model "$MODEL_ID" \
  --arg sha "$MODEL_SHA256" \
  --arg backend "$RUNTIME_COMPUTE_BACKEND" \
  --argjson cuda_active "$RUNTIME_CUDA_ACTIVE" '
  (.members | length) == 2 and
  all(.members[];
    .capabilities.runtime_model_id == $model and
    .capabilities.runtime_model_sha256 == $sha and
    .capabilities.runtime_engine == "llama.cpp" and
    .capabilities.runtime_compute_backend == $backend and
    .capabilities.runtime_cuda_active == $cuda_active and
    .capabilities.runtime_execution_mode == "pipeline_parallel")
' "$MEMBERS_FILE" >/dev/null

curl -fsS "http://127.0.0.1:$NODE0_PORT/v1/routes/preview?model=$MODEL_ID&stage_count=2&allow_colocated_stages=true" >"$PREVIEW_FILE"
jq -e '
  .valid == true and .mode == "pipeline_parallel" and
  .topology == "colocated" and .stage_count == 2 and
  .logical_node_count == 2 and .physical_host_count == 1
' "$PREVIEW_FILE" >/dev/null

COORDINATOR_NODE_ID="$(curl -fsS "http://127.0.0.1:$NODE0_PORT/healthz" | jq -r '.node_id')"
if [[ -z "$COORDINATOR_NODE_ID" || "$COORDINATOR_NODE_ID" == "null" ]]; then
  echo "coordinator node ID is unavailable" >&2
  exit 1
fi

curl -fsS -X POST "http://127.0.0.1:$NODE0_PORT/v1/runtime/generate" \
  -H 'Content-Type: application/json' \
  -H "X-JetsonFabric-Coordinator-Node-ID: $COORDINATOR_NODE_ID" \
  -H "X-JetsonFabric-Cluster-Token: $CLUSTER_TOKEN" \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg prompt "$RAW_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    --slurpfile preview "$PREVIEW_FILE" \
    '{request_id:"phase1-runtime-generation",session_id:"phase1-runtime-session",model_id:$model,prompt:$prompt,max_tokens:$max_tokens,stages:$preview[0].stages}')" \
  >"$RUNTIME_GENERATION_FILE"

jq -s -e \
  --argjson baseline "$BASELINE_TOKENS" \
  --arg finish_reason "$BASELINE_FINISH_REASON" \
  --argjson passes "$GENERATION_PASSES" \
  --argjson stage_count 2 '
  ([.[] | select(.type == "done")]) as $done |
  length == (($baseline | length) + 1) and
  all(.[]; .type == "token" or .type == "done") and
  ([.[] | select(.type == "token") | .token] == $baseline) and
  ([.[] | select(.type == "token") | .index] == [range(0; ($baseline | length))]) and
  ($done | length) == 1 and
  $done[0].finish_reason == $finish_reason and
  $done[0].completion_tokens == ($baseline | length) and
  $done[0].sampled_tokens == $baseline and
  $done[0].stage_calls == ($passes * $stage_count) and
  $done[0].remote_stage_calls == ($passes * ($stage_count - 1))
' "$RUNTIME_GENERATION_FILE" >/dev/null

curl -fsS -X POST "http://127.0.0.1:$NODE0_PORT/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg payload "$RAW_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{request_id:"phase1-diagnostic",model:$model,payload:$payload,max_tokens:$max_tokens,stage_count:2,allow_colocated_stages:true}')" \
  >"$DIAGNOSTIC_FILE"

jq -e \
  --arg sha "$MODEL_SHA256" \
  --argjson baseline "$BASELINE_TOKENS" \
  --arg finish_reason "$BASELINE_FINISH_REASON" \
  --argjson passes "$GENERATION_PASSES" '
  .runtime_identity.model_sha256 == $sha and
  .inter_stage_payload_kind == "activation" and
  .result.sampled_tokens == $baseline and
  .result.finish_reason == $finish_reason and
  .result.completion_tokens == ($baseline | length) and
  (.result.stages | length) == (2 * $passes) and
  .result.stages as $traces |
  all($traces[];
    if .payload_kind_out == "activation" then
      . as $source |
      any($traces[];
        .phase == $source.phase and
        .decode_step == $source.decode_step and
        .stage_index == ($source.stage_index + 1) and
        .payload_kind_in == "activation" and
        .payload_in == $source.payload_out and
        .payload_crc32_in == $source.payload_crc32_out)
    else true end)
' "$DIAGNOSTIC_FILE" >/dev/null

CHAT_METRICS="$(curl -sS -D "$CHAT_HEADERS" -o "$CHAT_FILE" -w '%{http_code} %{time_total}' \
  -X POST "http://127.0.0.1:$NODE1_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg prompt "$CHAT_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,jetsonfabric:{stage_count:2,allow_colocated_stages:true}}')")"
read -r CHAT_STATUS CHAT_SECONDS <<<"$CHAT_METRICS"
if [[ "$CHAT_STATUS" != "200" ]]; then
  echo "chat completion returned HTTP $CHAT_STATUS: $(cat "$CHAT_FILE")" >&2
  exit 1
fi
CHAT_DURATION_MS="$(awk -v seconds="$CHAT_SECONDS" 'BEGIN { printf "%.3f", seconds * 1000 }')"

STREAM_STATUS="$(curl -sS -N -o "$STREAM_FILE" -w '%{http_code}' -X POST "http://127.0.0.1:$NODE1_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg prompt "$CHAT_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,stream:true,jetsonfabric:{stage_count:2,allow_colocated_stages:true}}')")"
if [[ "$STREAM_STATUS" != "200" ]]; then
  echo "streaming chat returned HTTP $STREAM_STATUS: $(cat "$STREAM_FILE")" >&2
  exit 1
fi
sed -n 's/^data: //p' "$STREAM_FILE" | grep -v '^\[DONE\]$' >"$STREAM_EVENTS_FILE"

RUNTIME0_MEMORY="$(runtime_memory_for_port "$RUNTIME0_PORT" "$MODEL_PATH")"
RUNTIME1_MEMORY="$(runtime_memory_for_port "$RUNTIME1_PORT" "$MODEL_PATH")"
jq -e '.rss_bytes > 0 and .pss_bytes > 0 and .model_mapping_rss_bytes >= 0' <<<"$RUNTIME0_MEMORY" >/dev/null
jq -e '.rss_bytes > 0 and .pss_bytes > 0 and .model_mapping_rss_bytes >= 0' <<<"$RUNTIME1_MEMORY" >/dev/null

jq -e --arg model "$MODEL_ID" '
  .object == "chat.completion" and .model == $model and
  (.choices | length) == 1 and .choices[0].message.role == "assistant" and
  (.choices[0].finish_reason == "stop" or .choices[0].finish_reason == "length") and
  (.usage.completion_tokens | type) == "number" and
  .usage.completion_tokens >= 0 and
  (.usage.completion_tokens | floor) == .usage.completion_tokens
' "$CHAT_FILE" >/dev/null
CHAT_FINISH_REASON="$(jq -r '.choices[0].finish_reason' "$CHAT_FILE")"
CHAT_COMPLETION_TOKENS="$(jq -r '.usage.completion_tokens' "$CHAT_FILE")"
if [[ -n "$EXPECTED_CHAT_FINISH_REASON" && "$CHAT_FINISH_REASON" != "$EXPECTED_CHAT_FINISH_REASON" ]]; then
  echo "chat finish reason does not match JF_EXPECTED_CHAT_FINISH_REASON: expected=$EXPECTED_CHAT_FINISH_REASON actual=$CHAT_FINISH_REASON" >&2
  exit 1
fi
if [[ -n "$EXPECTED_CHAT_COMPLETION_TOKENS" && "$CHAT_COMPLETION_TOKENS" != "$EXPECTED_CHAT_COMPLETION_TOKENS" ]]; then
  echo "chat completion tokens do not match JF_EXPECTED_CHAT_COMPLETION_TOKENS: expected=$EXPECTED_CHAT_COMPLETION_TOKENS actual=$CHAT_COMPLETION_TOKENS" >&2
  exit 1
fi
grep -qi "^X-JetsonFabric-Model-SHA256: $MODEL_SHA256" "$CHAT_HEADERS"
grep -qi '^X-JetsonFabric-Topology: colocated' "$CHAT_HEADERS"
grep -qi '^X-JetsonFabric-Generation-Owner: runtime' "$CHAT_HEADERS"
grep -qi '^X-JetsonFabric-Pipeline-Leader: ' "$CHAT_HEADERS"
STAGE_CALLS="$(awk -F': ' 'tolower($1) == "x-jetsonfabric-stage-calls" {gsub("\r", "", $2); print $2}' "$CHAT_HEADERS")"
REMOTE_STAGE_CALLS="$(awk -F': ' 'tolower($1) == "x-jetsonfabric-remote-stage-calls" {gsub("\r", "", $2); print $2}' "$CHAT_HEADERS")"
CHAT_GENERATION_PASSES="$CHAT_COMPLETION_TOKENS"
if [[ "$CHAT_FINISH_REASON" == "stop" ]]; then
  CHAT_GENERATION_PASSES=$((CHAT_GENERATION_PASSES + 1))
fi
if [[ ! "$STAGE_CALLS" =~ ^[0-9]+$ || ! "$REMOTE_STAGE_CALLS" =~ ^[0-9]+$ ||
      "$STAGE_CALLS" -ne $((CHAT_GENERATION_PASSES * 2)) ||
      "$REMOTE_STAGE_CALLS" -ne "$CHAT_GENERATION_PASSES" ]]; then
  echo "invalid runtime-owned stage call evidence: stage_calls=$STAGE_CALLS remote_stage_calls=$REMOTE_STAGE_CALLS" >&2
  exit 1
fi
jq -s -e '
  length >= 3 and
  .[0].object == "chat.completion.chunk" and
  .[0].choices[0].delta.role == "assistant" and
  any(.[]; (.choices[0].delta.content // "") != "") and
  ([.[] | select(.choices[0].finish_reason != null)] | length) == 1 and
  any(.[]; .choices[0].finish_reason == "length" or .choices[0].finish_reason == "stop")
' "$STREAM_EVENTS_FILE" >/dev/null
grep -q '^data: \[DONE\]$' "$STREAM_FILE"
STREAM_FINISH_REASON="$(jq -rs '[.[] | .choices[0].finish_reason // empty] | last' "$STREAM_EVENTS_FILE")"
if [[ "$STREAM_FINISH_REASON" != "$CHAT_FINISH_REASON" ]]; then
  echo "buffered and streaming finish reasons differ: buffered=$CHAT_FINISH_REASON streaming=$STREAM_FINISH_REASON" >&2
  exit 1
fi
CHAT_TEXT="$(jq -r '.choices[0].message.content' "$CHAT_FILE")"
STREAM_TEXT="$(jq -rs 'map(.choices[0].delta.content // "") | join("")' "$STREAM_EVENTS_FILE")"
if [[ "$CHAT_TEXT" != "$STREAM_TEXT" ]]; then
  echo "buffered and streaming chat text differ: buffered=$(printf '%q' "$CHAT_TEXT") streaming=$(printf '%q' "$STREAM_TEXT")" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -n \
  --arg timestamp "$TIMESTAMP" \
  --arg model_id "$MODEL_ID" \
  --arg model_path "$MODEL_PATH" \
  --arg model_sha256 "$MODEL_SHA256" \
  --arg raw_prompt "$RAW_PROMPT" \
  --arg baseline_finish_reason "$BASELINE_FINISH_REASON" \
  --arg chat_prompt "$CHAT_PROMPT" \
  --arg chat_duration_ms "$CHAT_DURATION_MS" \
  --argjson runtime0_memory "$RUNTIME0_MEMORY" \
  --argjson runtime1_memory "$RUNTIME1_MEMORY" \
  --argjson layer_count "$LAYER_COUNT" \
  --argjson split_layer "$SPLIT_LAYER" \
  --argjson baseline_tokens "$BASELINE_TOKENS" \
  --slurpfile members "$MEMBERS_FILE" \
  --slurpfile runtime0 "$RUNTIME0_STATUS_FILE" \
  --slurpfile runtime1 "$RUNTIME1_STATUS_FILE" \
  --slurpfile preview "$PREVIEW_FILE" \
  --slurpfile diagnostic "$DIAGNOSTIC_FILE" \
  --slurpfile runtime_generation "$RUNTIME_GENERATION_FILE" \
  --slurpfile chat "$CHAT_FILE" \
  --rawfile chat_stream "$STREAM_FILE" \
  --argjson stage_calls "$STAGE_CALLS" \
  --argjson remote_stage_calls "$REMOTE_STAGE_CALLS" \
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
    model_residency: [$runtime0[0].model_memory, $runtime1[0].model_memory],
    process_memory: [$runtime0_memory, $runtime1_memory],
    correctness: {
      raw_prompt: $raw_prompt,
      baseline_tokens: $baseline_tokens,
      baseline_finish_reason: $baseline_finish_reason,
      sampled_tokens: $diagnostic[0].result.sampled_tokens,
      runtime_generation_events: $runtime_generation,
      runtime_generation_sampled_tokens: [$runtime_generation[] | select(.type == "token") | .token],
      activation_crc_continuity: true,
      chat_prompt: $chat_prompt,
      chat_response: $chat[0],
      streaming_response: $chat_stream
    },
    performance: {
      chat_duration_ms: ($chat_duration_ms | tonumber),
      stage_trace: $diagnostic[0].result.stages,
      runtime_owned_stage_calls: $stage_calls,
      runtime_owned_remote_stage_calls: $remote_stage_calls
    }
  }' >"$REPORT_PATH"

echo "Phase 1 colocated validation passed"
echo "Model: $MODEL_ID"
echo "Model SHA-256: $MODEL_SHA256"
echo "Layers: [0,$SPLIT_LAYER) -> [$SPLIT_LAYER,$LAYER_COUNT)"
echo "Ports: stage0 node/runtime $NODE0_PORT/$RUNTIME0_PORT; stage1 node/runtime $NODE1_PORT/$RUNTIME1_PORT"
echo "Baseline tokens: $BASELINE_TOKENS"
echo "Baseline finish reason: $BASELINE_FINISH_REASON"
echo "Report: $REPORT_PATH"
