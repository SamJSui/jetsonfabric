#!/usr/bin/env sh
set -eu

control_url=http://127.0.0.1:52415
request=examples/p0-local-smoke/chat-request.json
count=5
concurrency=1
output=data/desktop-chat-benchmark.json

while [ "$#" -gt 0 ]; do
  case "$1" in
    --control-url)
      control_url=$2
      shift 2
      ;;
    --request)
      request=$2
      shift 2
      ;;
    --count)
      count=$2
      shift 2
      ;;
    --concurrency)
      concurrency=$2
      shift 2
      ;;
    --output)
      output=$2
      shift 2
      ;;
    --help)
      printf 'usage: %s [--control-url URL] [--request PATH] [--count N] [--concurrency N] [--output PATH]\n' "$0"
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

mkdir -p "$go_cache"
export GOCACHE="$go_cache"

cd "$repo_root"

"$go_bin" run ./cmd/jetsonfabric-bench \
  --url "$control_url/v1/chat/completions" \
  --request "$request" \
  --count "$count" \
  --concurrency "$concurrency" \
  --output "$output"

printf 'benchmark summary: %s\n' "$output"
