#!/usr/bin/env sh
set -eu

count=3
start_port=52416
control_url=http://127.0.0.1:52415
join_token=dev-token
node_prefix=desktop-agent
llama_url=http://127.0.0.1:8080
llama_models=qwen2.5-coder-1.5b-q4
host=127.0.0.1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --count)
      count=$2
      shift 2
      ;;
    --start-port)
      start_port=$2
      shift 2
      ;;
    --control-url)
      control_url=$2
      shift 2
      ;;
    --join-token)
      join_token=$2
      shift 2
      ;;
    --node-prefix)
      node_prefix=$2
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
    --host)
      host=$2
      shift 2
      ;;
    --help)
      printf 'usage: %s [--count N] [--start-port PORT] [--control-url URL] [--join-token TOKEN] [--node-prefix PREFIX] [--llama-url URL] [--llama-models CSV] [--host HOST]\n' "$0"
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

i=1
while [ "$i" -le "$count" ]; do
  port=$((start_port + i - 1))
  node_id="$node_prefix-$i"
  advertise_url="http://$host:$port"
  sh "$script_dir/run-agent.sh" \
    --control-url "$control_url" \
    --join-token "$join_token" \
    --node-id "$node_id" \
    --host "$host" \
    --port "$port" \
    --advertise-url "$advertise_url" \
    --llama-url "$llama_url" \
    --llama-models "$llama_models" \
    --background
  i=$((i + 1))
done

printf '\nDesktop agents are simulation nodes. They can test discovery, routing, planning, and proxy overhead, but they do not execute real distributed layers yet.\n'
printf 'Layer split plan: curl -sS "%s/v1/layer-split/plan?model=%s"\n' "$control_url" "$llama_models"
