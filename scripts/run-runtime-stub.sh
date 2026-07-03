#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)

listen=${LISTEN:-127.0.0.1:9090}
model=${MODEL:-qwen2.5-coder-1.5b-q4}
mode=${MODE:-single_node}

exec "$repo_root/dist/jetsonfabric-runtime-worker" \
  --listen "$listen" \
  --model "$model" \
  --mode "$mode"
