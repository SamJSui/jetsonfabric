#!/usr/bin/env sh
set -eu

control_url=http://127.0.0.1:52415
join_token=dev-token
node_id=dev-node
host=127.0.0.1
port=52416
advertise_url=http://127.0.0.1:52416
model_artifacts=configs/model-artifacts.example.json
llama_url=
llama_models=
once=
background=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --control-url)
      control_url=$2
      shift 2
      ;;
    --join-token)
      join_token=$2
      shift 2
      ;;
    --node-id)
      node_id=$2
      shift 2
      ;;
    --host)
      host=$2
      shift 2
      ;;
    --port)
      port=$2
      shift 2
      ;;
    --advertise-url)
      advertise_url=$2
      shift 2
      ;;
    --model-artifacts)
      model_artifacts=$2
      shift 2
      ;;
    --llama-url)
      llama_url=$2
      shift 2
      ;;
    --llama-models)
      llama_models=$2
      shift 2
      ;;
    --once)
      once=--once
      shift
      ;;
    --background)
      background=1
      shift
      ;;
    --help)
      printf 'usage: %s [--control-url URL] [--join-token TOKEN] [--node-id ID] [--host HOST] [--port PORT] [--advertise-url URL] [--model-artifacts PATH] [--llama-url URL] [--llama-models CSV] [--once] [--background]\n' "$0"
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
go_bin=${GO:-go}
go_cache="$repo_root/.cache/go-build"
log_path="$repo_root/.cache/logs/agent-$node_id.log"
pid_path="$repo_root/.cache/pids/agent-$node_id.pid"

mkdir -p "$go_cache" "$(dirname "$log_path")" "$(dirname "$pid_path")"
export GOCACHE="$go_cache"

set -- run ./cmd/jetsonfabric-agent \
  --control-url "$control_url" \
  --join-token "$join_token" \
  --node-id "$node_id" \
  --host "$host" \
  --port "$port" \
  --advertise-url "$advertise_url" \
  --model-artifacts "$model_artifacts"

if [ -n "$llama_url" ]; then
  set -- "$@" --llama-url "$llama_url"
fi

if [ -n "$llama_models" ]; then
  set -- "$@" --llama-models "$llama_models"
fi

if [ -n "$once" ]; then
  set -- "$@" "$once"
fi

cd "$repo_root"

if [ -n "$background" ]; then
  nohup "$go_bin" "$@" > "$log_path" 2>&1 &
  printf '%s\n' "$!" > "$pid_path"
  printf 'jetsonfabric-agent started with pid %s\n' "$(cat "$pid_path")"
  printf 'proxy: %s\n' "$advertise_url"
  printf 'log: %s\n' "$log_path"
  exit 0
fi

exec "$go_bin" "$@"
