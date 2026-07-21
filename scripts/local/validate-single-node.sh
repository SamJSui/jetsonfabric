#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GGUF to validate}"
MODEL_ID="${MODEL_ID:-qwen2.5-coder-1.5b-q4}"
RAW_PROMPT="${JF_RAW_PROMPT:-Once upon a time}"
CHAT_PROMPT="${JF_CHAT_PROMPT:-Hi}"
MAX_TOKENS="${JF_MAX_TOKENS:-4}"
EXPECTED_TOKENS="${JF_EXPECTED_TOKENS:-}"
NODE_PORT="${JF_NODE0_PORT:-19180}"
RUNTIME_BUILD_DIR="${RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-single-cpu}"
RUNTIME_BIN="${RUNTIME_BIN:-$ROOT_DIR/dist/jetsonfabric-runtime-worker-single-cpu}"
NODE_BIN="${NODE_BIN:-$ROOT_DIR/dist/jetsonfabric-node}"
WORK_DIR="$(mktemp -d)"
LOG_DIR="$WORK_DIR/logs"
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
    echo "Single-node pipeline validation failed." >&2
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

NODE_URL="http://127.0.0.1:$NODE_PORT"
"$NODE_BIN" \
  --cluster-id single-node-integration \
  --node-name single-stage \
  --listen "127.0.0.1:$NODE_PORT" \
  --advertise-url "$NODE_URL" \
  --data-dir "$WORK_DIR/node" \
  --runtime-url auto \
  --runtime-bin "$RUNTIME_BIN" \
  --runtime-listen 127.0.0.1:0 \
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
  >"$LOG_DIR/single-stage.log" 2>&1 &
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

DIAGNOSTIC_FILE="$WORK_DIR/diagnostic.json"
HTTP_CODE="$(curl -sS -o "$DIAGNOSTIC_FILE" -w '%{http_code}' \
  -X POST "$NODE_URL/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg payload "$RAW_PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{request_id:"single-node-integration",model:$model,payload:$payload,max_tokens:$max_tokens,stage_count:1}')")"
if [[ "$HTTP_CODE" != "200" ]]; then
  echo "single-node diagnostic returned HTTP $HTTP_CODE" >&2
  jq . "$DIAGNOSTIC_FILE" >&2 2>/dev/null || cat "$DIAGNOSTIC_FILE" >&2
  exit 1
fi
jq -e --argjson baseline "$BASELINE_TOKENS" --argjson max_tokens "$MAX_TOKENS" '
  .plan.valid == true and
  .plan.mode == "pipeline_parallel" and
  .plan.stage_count == 1 and
  (.plan.stages | length) == 1 and
  .plan.stages[0].stage_index == 0 and
  .plan.stages[0].stage_count == 1 and
  .result.payload_kind == "sampled_token" and
  .result.sampled_tokens == $baseline and
  .result.prompt_tokens > 0 and
  (.result.stages | length) == $max_tokens and
  .result.stages[0].phase == "prefill" and
  all(.result.stages[1:][]; .phase == "decode")
' "$DIAGNOSTIC_FILE" >/dev/null

CHAT_FILE="$WORK_DIR/chat.json"
CHAT_CODE="$(curl -sS -o "$CHAT_FILE" -w '%{http_code}' \
  -X POST "$NODE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "$(jq -nc \
    --arg model "$MODEL_ID" \
    --arg prompt "$CHAT_PROMPT" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:1}')")"
if [[ "$CHAT_CODE" != "200" ]]; then
  echo "single-node chat returned HTTP $CHAT_CODE" >&2
  jq . "$CHAT_FILE" >&2 2>/dev/null || cat "$CHAT_FILE" >&2
  exit 1
fi
jq -e --arg model "$MODEL_ID" '
  .object == "chat.completion" and
  .model == $model and
  (.choices | length) == 1 and
  .choices[0].message.role == "assistant" and
  .usage.prompt_tokens > 0
' "$CHAT_FILE" >/dev/null

echo "Single-node pipeline validation passed"
echo "Model: $MODEL_ID"
echo "Layers: [0,$LAYER_COUNT)"
echo "Baseline tokens: $BASELINE_TOKENS"
