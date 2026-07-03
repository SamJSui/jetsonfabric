#!/usr/bin/env sh
set -eu

control_url=${CONTROL_URL:-http://127.0.0.1:52415}
agent_url=${AGENT_URL:-}
node_name=${NODE_NAME:-dopey}
model=${MODEL:-qwen2.5-coder-1.5b-q4}
request_path=${REQUEST_PATH:-examples/poc-local-smoke/chat-request.json}
benchmarks_path=${BENCHMARKS_PATH:-data/benchmarks.jsonl}
expected_route_mode=${EXPECTED_ROUTE_MODE:-single_node}
expected_engine=${EXPECTED_ENGINE:-}
skip_agent_health=${SKIP_AGENT_HEALTH:-}

usage() {
  cat <<USAGE
usage: $0 [options]

Options:
  --control-url URL
  --agent-url URL
  --node-name NAME
  --model MODEL_ID
  --request PATH
  --benchmarks PATH
  --expected-route-mode MODE
  --expected-engine ENGINE
  --skip-agent-health
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --control-url) control_url=$2; shift 2 ;;
    --agent-url) agent_url=$2; shift 2 ;;
    --node-name) node_name=$2; shift 2 ;;
    --model) model=$2; shift 2 ;;
    --request) request_path=$2; shift 2 ;;
    --benchmarks) benchmarks_path=$2; shift 2 ;;
    --expected-route-mode) expected_route_mode=$2; shift 2 ;;
    --expected-engine) expected_engine=$2; shift 2 ;;
    --skip-agent-health) skip_agent_health=1; shift ;;
    --help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

check_json_with_jq() {
  file=$1
  message=$2
  shift 2
  if jq -e "$@" "$file" >/dev/null; then
    printf 'OK: %s\n' "$message"
  else
    printf 'FAIL: %s\n' "$message" >&2
    printf 'response file: %s\n' "$file" >&2
    jq . "$file" >&2 || true
    exit 1
  fi
}

http_get() {
  url=$1
  output=$2
  printf 'GET %s\n' "$url"
  curl -fsS "$url" -o "$output"
}

require_command curl
require_command jq

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
work_dir="$repo_root/.cache/smoke-node"
mkdir -p "$work_dir"

case "$request_path" in
  /*) request_file=$request_path ;;
  *) request_file="$repo_root/$request_path" ;;
esac

case "$benchmarks_path" in
  /*) benchmarks_file=$benchmarks_path ;;
  *) benchmarks_file="$repo_root/$benchmarks_path" ;;
esac

if [ ! -f "$request_file" ]; then
  printf 'missing request file: %s\n' "$request_file" >&2
  exit 1
fi

printf 'JetsonFabric node smoke test\n'
printf 'control_url: %s\n' "$control_url"
printf 'agent_url: %s\n' "${agent_url:-<skipped>}"
printf 'node_name: %s\n' "$node_name"
printf 'model: %s\n' "$model"
printf 'expected_route_mode: %s\n' "$expected_route_mode"
if [ -n "$expected_engine" ]; then
  printf 'expected_engine: %s\n' "$expected_engine"
fi

control_health="$work_dir/control-health.json"
http_get "$control_url/healthz" "$control_health"
check_json_with_jq "$control_health" 'control plane health is ok' '.status == "ok"'

if [ -z "$skip_agent_health" ] && [ -n "$agent_url" ]; then
  agent_health="$work_dir/agent-health.json"
  http_get "$agent_url/healthz" "$agent_health"
  check_json_with_jq "$agent_health" 'agent proxy health is ok' '.status == "ok"'
fi

nodes_json="$work_dir/nodes.json"
http_get "$control_url/v1/nodes" "$nodes_json"
check_json_with_jq "$nodes_json" "node $node_name is registered" \
  --arg node "$node_name" '.nodes[]? | select(.node_name == $node)'

preview_json="$work_dir/route-preview.json"
http_get "$control_url/v1/routes/preview?model=$model" "$preview_json"
check_json_with_jq "$preview_json" "route preview marks $node_name valid for $model" \
  --arg node "$node_name" '.placements[]? | select(.node_name == $node and .valid == true)'

chat_json="$work_dir/chat-response.json"
printf 'POST %s/v1/chat/completions\n' "$control_url"
curl -fsS -X POST "$control_url/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$request_file" \
  -o "$chat_json"

check_json_with_jq "$chat_json" 'chat response includes at least one choice' '.choices | length > 0'
check_json_with_jq "$chat_json" "chat response route node is $node_name" \
  --arg node "$node_name" '.jetsonfabric_route.node_name == $node'
check_json_with_jq "$chat_json" "chat response route mode is $expected_route_mode" \
  --arg mode "$expected_route_mode" '.jetsonfabric_route.mode == $mode'

if [ -n "$expected_engine" ]; then
  check_json_with_jq "$chat_json" "chat response engine is $expected_engine" \
    --arg engine "$expected_engine" '(.jetsonfabric_route.engine // .jetsonfabric_route.backend_kind) == $engine'
fi

printf '\nassistant response:\n'
jq -r '.choices[0].message.content' "$chat_json"

printf '\nroute metadata:\n'
jq '.jetsonfabric_route' "$chat_json"

if [ -f "$benchmarks_file" ]; then
  if jq -e --arg node "$node_name" --arg model "$model" '
    select(.node_name == $node and .model_id == $model)
  ' "$benchmarks_file" >/dev/null 2>&1; then
    printf 'OK: benchmark file contains structured record for %s and %s\n' "$node_name" "$model"
  else
    printf 'WARN: benchmark file exists but did not find node/model record: %s\n' "$benchmarks_file" >&2
  fi
else
  printf 'WARN: benchmark file not found yet: %s\n' "$benchmarks_file" >&2
fi

printf '\nSmoke test complete. Artifacts are in %s\n' "$work_dir"