#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
BIN_DIR="$WORK_DIR/bin"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$BIN_DIR" "$LOG_DIR"

MODEL_PATH="${CI_MODEL_PATH:?CI_MODEL_PATH is required}"
RUNTIME_BUILD_DIR="${CI_RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-ci-cpu}"
RUNTIME_BIN="$RUNTIME_BUILD_DIR/jetsonfabric-runtime-worker"
NODE0_PORT="${CI_ACTIVATION_NODE0_PORT:-18280}"
NODE1_PORT="${CI_ACTIVATION_NODE1_PORT:-18281}"
CLUSTER_TOKEN="${JF_CLUSTER_TOKEN:-jetsonfabric-ci-activation-token}"

PIDS=()
cleanup() {
  local status=$?
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "Binary activation E2E failed. Logs:" >&2
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
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.25
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

wait_for_two_members() {
  local url=$1
  for _ in $(seq 1 240); do
    if [[ "$(curl -fsS "$url" 2>/dev/null | jq '.members | length' 2>/dev/null || echo 0)" == "2" ]]; then return 0; fi
    sleep 0.25
  done
  echo "Timed out waiting for two members at $url" >&2
  return 1
}

cd "$ROOT_DIR"
if [[ ! -x "$RUNTIME_BIN" ]]; then
  echo "Runtime binary missing: $RUNTIME_BIN" >&2
  exit 1
fi

go build -buildvcs=false -o "$BIN_DIR/jetsonfabric-node" ./cmd/jetsonfabric-node

MODELS_PATH="$WORK_DIR/models.json"
cat >"$MODELS_PATH" <<'JSON'
{
  "models": [
    {
      "id": "synthetic-activation-v1",
      "family": "synthetic",
      "supported_engines": ["synthetic"],
      "layer_count": 2,
      "min_memory_gb": 0,
      "preferred_compute": "cpu",
      "placement_modes": ["pipeline_parallel"]
    }
  ]
}
JSON

start_node() {
  local name=$1 port=$2 peer_port=$3 stage_index=$4 layer_start=$5 layer_end=$6 role=$7
  JETSONFABRIC_CLUSTER_TOKEN="$CLUSTER_TOKEN" "$BIN_DIR/jetsonfabric-node" \
    --cluster-id ci-binary-activation \
    --node-name "$name" \
    --listen "127.0.0.1:$port" \
    --advertise-url "http://127.0.0.1:$port" \
    --data-dir "$WORK_DIR/$name" \
    --runtime-url auto \
    --runtime-bin "$RUNTIME_BIN" \
    --runtime-listen 127.0.0.1:0 \
    --runtime-compute-backend cpu \
    --runtime-mode pipeline_parallel \
    --engine synthetic \
    --model synthetic-activation-v1 \
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
    --models "$MODELS_PATH" \
    >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
}

start_node ci-activation-stage0 "$NODE0_PORT" "$NODE1_PORT" 0 0 1 coordinator
start_node ci-activation-stage1 "$NODE1_PORT" "$NODE0_PORT" 1 1 2 worker

wait_for_url "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"
wait_for_url "http://127.0.0.1:$NODE1_PORT/v1/cluster/members"
wait_for_two_members "http://127.0.0.1:$NODE0_PORT/v1/cluster/members"

response_file="$WORK_DIR/response.json"
http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$NODE0_PORT/v1/layer-split/run" \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "ci-binary-activation",
    "model": "synthetic-activation-v1",
    "payload": "activation-checkpoint",
    "max_tokens": 1,
    "stage_count": 2,
    "allow_colocated_stages": true
  }')"

if [[ "$http_code" != "200" ]]; then
  echo "Expected HTTP 200, got $http_code" >&2
  cat "$response_file" >&2
  exit 1
fi

jq -e '
  .inter_stage_payload_kind == "activation" and
  .plan.mode == "pipeline_parallel" and
  .plan.stage_count == 2 and
  .result.payload_kind == "sampled_token" and
  (.result.sampled_token | type == "number") and
  .result.payload_bytes == 4 and
  (.result.stages | length) == 2 and
  .result.stages[0].payload_kind_in == "text" and
  .result.stages[0].payload_kind_out == "activation" and
  .result.stages[0].payload_out == 256 and
  .result.stages[0].transport == "http_binary_v1" and
  .result.stages[1].payload_kind_in == "activation" and
  .result.stages[1].payload_kind_out == "sampled_token" and
  .result.stages[1].payload_in == 256 and
  .result.stages[1].transport == "http_binary_v1" and
  .result.stages[0].payload_crc32_out == .result.stages[1].payload_crc32_in
' "$response_file" >/dev/null

echo "Two-node binary activation E2E passed"
jq '{inter_stage_payload_kind, sampled_token: .result.sampled_token, stages: .result.stages}' "$response_file"
