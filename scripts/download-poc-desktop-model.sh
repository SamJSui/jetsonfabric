#!/usr/bin/env sh
set -eu

force=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      force=1
      shift
      ;;
    --help)
      printf 'usage: %s [--force]\n' "$0"
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
downloads_dir="$repo_root/.cache/downloads"
tools_dir="$repo_root/.cache/tools"

llama_release=b9821
llama_archive=llama-b9821-bin-ubuntu-x64.tar.gz
llama_dir="$tools_dir/llama-$llama_release"
llama_url="https://github.com/ggml-org/llama.cpp/releases/download/$llama_release/$llama_archive"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

download_file() {
  url=$1
  output=$2

  if [ -f "$output" ] && [ -z "$force" ]; then
    printf 'exists: %s\n' "$output"
    return
  fi

  printf 'downloading: %s\n' "$url"
  curl -L --fail --show-error "$url" -o "$output"
}

require_command curl
require_command tar

mkdir -p "$downloads_dir" "$tools_dir"

download_file "$llama_url" "$downloads_dir/$llama_archive"
if [ ! -x "$llama_dir/llama-server" ] || [ -n "$force" ]; then
  tar -xzf "$downloads_dir/$llama_archive" -C "$tools_dir"
fi

set -- sh "$script_dir/download-model-artifact.sh" --model-id qwen2.5-coder-1.5b-q4
if [ -n "$force" ]; then
  set -- "$@" --force
fi
"$@"

printf 'llama-server: %s\n' "$llama_dir/llama-server"
