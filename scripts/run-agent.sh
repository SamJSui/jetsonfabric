#!/usr/bin/env sh
set -eu

control_url=http://127.0.0.1:52415
join_token=dev-token
node_name=
listen=127.0.0.1:52416
advertise_url=http://127.0.0.1:52416
model_artifacts=configs/model-artifacts.example.json
llama_url=
model=
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
    --node-name)
      node_name=$2
      shift 2
      ;;
    --listen)
      listen=$2
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
    --model)
      model=$2
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
      printf 'usage: %s [--control-url URL] [--join-token TOKEN] [--node-name NAME] [--listen ADDR] [--advertise-url URL] [--model-artifacts PATH] [--llama-url URL] [--model MODEL_ID] [--once] [--background]\n' "$0"
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
log_node_name=$node_name
if [ -z "$log_node_name" ]; then
  log_node_name=$(hostname)
fi
log_path="$repo_root/.cache/logs/agent-$log_node_name.log"
pid_path="$repo_root/.cache/pids/agent-$log_node_name.pid"

mkdir -p "$go_cache" "$(dirname "$log_path")" "$(dirname "$pid_path")"
export GOCACHE="$go_cache"

set -- run ./cmd/jetsonfabric-agent \
  --control-url "$control_url" \
  --join-token "$join_token" \
  --listen "$listen" \
  --advertise-url "$advertise_url" \
  --model-artifacts "$model_artifacts"

if [ -n "$node_name" ]; then
  set -- "$@" --node-name "$node_name"
fi

if [ -n "$llama_url" ]; then
  set -- "$@" --llama-url "$llama_url"
fi

if [ -n "$model" ]; then
  set -- "$@" --model "$model"
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
