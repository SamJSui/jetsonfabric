#!/usr/bin/env sh
set -eu

count=3
node_prefix=desktop-agent

while [ "$#" -gt 0 ]; do
  case "$1" in
    --count)
      count=$2
      shift 2
      ;;
    --node-prefix)
      node_prefix=$2
      shift 2
      ;;
    --help)
      printf 'usage: %s [--count N] [--node-prefix PREFIX]\n' "$0"
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
pid_dir="$repo_root/.cache/pids"

i=1
while [ "$i" -le "$count" ]; do
  node_name="$node_prefix-$i"
  pid_path="$pid_dir/agent-$node_name.pid"
  if [ ! -f "$pid_path" ]; then
    printf 'no pid file for %s\n' "$node_name"
    i=$((i + 1))
    continue
  fi

  pid=$(cat "$pid_path")
  children=$(pgrep -P "$pid" 2>/dev/null || true)
  if [ -n "$children" ]; then
    kill $children 2>/dev/null || true
  fi
  kill "$pid" 2>/dev/null || true
  rm -f "$pid_path"
  printf 'stopped %s\n' "$node_name"
  i=$((i + 1))
done
