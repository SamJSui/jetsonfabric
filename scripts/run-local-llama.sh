#!/usr/bin/env sh
set -eu

host=127.0.0.1
port=8080
model_alias=qwen2.5-coder-1.5b-q4
context_size=2048
background=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      host=$2
      shift 2
      ;;
    --port)
      port=$2
      shift 2
      ;;
    --model-alias)
      model_alias=$2
      shift 2
      ;;
    --ctx-size)
      context_size=$2
      shift 2
      ;;
    --background)
      background=1
      shift
      ;;
    --help)
      printf 'usage: %s [--host HOST] [--port PORT] [--model-alias ID] [--ctx-size TOKENS] [--background]\n' "$0"
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
llama_server="$repo_root/.cache/tools/llama-b9821/llama-server"
model_path="$repo_root/.cache/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
log_path="$repo_root/.cache/logs/llama-server.log"
pid_path="$repo_root/.cache/pids/llama-server.pid"

if [ ! -x "$llama_server" ]; then
  printf 'missing llama-server binary: %s\n' "$llama_server" >&2
  exit 1
fi

if [ ! -f "$model_path" ]; then
  printf 'missing model file: %s\n' "$model_path" >&2
  exit 1
fi

mkdir -p "$(dirname "$log_path")" "$(dirname "$pid_path")"

set -- "$llama_server" \
  --model "$model_path" \
  --alias "$model_alias" \
  --host "$host" \
  --port "$port" \
  --ctx-size "$context_size" \
  --no-webui

if [ -n "$background" ]; then
  nohup "$@" > "$log_path" 2>&1 &
  printf '%s\n' "$!" > "$pid_path"
  printf 'llama-server started on http://%s:%s with pid %s\n' "$host" "$port" "$(cat "$pid_path")"
  printf 'log: %s\n' "$log_path"
  exit 0
fi

exec "$@"
