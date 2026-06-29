#!/usr/bin/env sh
set -eu

listen=127.0.0.1:52415
join_token=dev-token
benchmarks_path=data/benchmarks.jsonl
background=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --listen)
      listen=$2
      shift 2
      ;;
    --join-token)
      join_token=$2
      shift 2
      ;;
    --benchmarks)
      benchmarks_path=$2
      shift 2
      ;;
    --background)
      background=1
      shift
      ;;
    --help)
      printf 'usage: %s [--listen ADDR] [--join-token TOKEN] [--benchmarks PATH] [--background]\n' "$0"
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
log_path="$repo_root/.cache/logs/control.log"
pid_path="$repo_root/.cache/pids/control.pid"

mkdir -p "$go_cache" "$(dirname "$log_path")" "$(dirname "$pid_path")"
export GOCACHE="$go_cache"

cd "$repo_root"

set -- "$go_bin" run ./cmd/jetsonfabric-control \
  --listen "$listen" \
  --join-token "$join_token" \
  --benchmarks "$benchmarks_path"

if [ -n "$background" ]; then
  nohup "$@" > "$log_path" 2>&1 &
  printf '%s\n' "$!" > "$pid_path"
  printf 'jetsonfabric-control started on http://%s with pid %s\n' "$listen" "$(cat "$pid_path")"
  printf 'log: %s\n' "$log_path"
  exit 0
fi

exec "$@"
