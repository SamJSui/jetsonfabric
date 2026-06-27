#!/usr/bin/env sh
set -eu

catalog_path=configs/model-artifacts.example.json
model_id=qwen2.5-coder-1.5b-q4
force=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --catalog)
      catalog_path=$2
      shift 2
      ;;
    --model-id)
      model_id=$2
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --help)
      printf 'usage: %s [--catalog PATH] [--model-id ID] [--force]\n' "$0"
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
catalog_file="$repo_root/$catalog_path"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

require_command curl
require_command jq

if [ ! -f "$catalog_file" ]; then
  printf 'missing model artifact catalog: %s\n' "$catalog_file" >&2
  exit 1
fi

artifact_json=$(jq -r --arg model_id "$model_id" '.artifacts[] | select(.model_id == $model_id)' "$catalog_file")
if [ -z "$artifact_json" ]; then
  printf 'model artifact not found: %s\n' "$model_id" >&2
  exit 1
fi

source_url=$(printf '%s\n' "$artifact_json" | jq -r '.source_url')
local_path=$(printf '%s\n' "$artifact_json" | jq -r '.local_path')
output_path="$repo_root/$local_path"

if [ -f "$output_path" ] && [ -z "$force" ]; then
  printf 'exists: %s\n' "$output_path"
  exit 0
fi

mkdir -p "$(dirname "$output_path")"
printf 'downloading model artifact %s\n' "$model_id"
printf 'source: %s\n' "$source_url"
printf 'output: %s\n' "$output_path"
curl -L --fail --show-error "$source_url" -o "$output_path"
