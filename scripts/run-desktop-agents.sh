#!/usr/bin/env sh
set -eu

count=3
start_port=52416
control_url=http://127.0.0.1:52415
join_token=dev-token
node_prefix=desktop-agent
llama_url=http://127.0.0.1:8080
model=qwen2.5-coder-1.5b-q4
listen_host=127.0.0.1

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
    --model)
      model=$2
      shift 2
      ;;
    --listen-host)
      listen_host=$2
      shift 2
      ;;
    --help)
      printf 'usage: %s [--count N] [--start-port PORT] [--control-url URL] [--join-token TOKEN] [--node-prefix PREFIX] [--llama-url URL] [--model MODEL_ID] [--listen-host HOST]\n' "$0"
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
  node_name="$node_prefix-$i"
  listen="$listen_host:$port"
  advertise_url="http://$listen_host:$port"
  sh "$script_dir/run-agent.sh" \
    --control-url "$control_url" \
    --join-token "$join_token" \
    --node-name "$node_name" \
    --listen "$listen" \
    --advertise-url "$advertise_url" \
    --llama-url "$llama_url" \
    --model "$model" \
    --background
  i=$((i + 1))
done

printf '\nDesktop agents are simulation nodes. They can test discovery, routing, planning, and proxy overhead, but they do not execute real distributed layers yet.\n'
printf 'Layer split plan: curl -sS "%s/v1/layer-split/plan?model=%s"\n' "$control_url" "$model"
