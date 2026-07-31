#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_CPP_DIR="${1:-$ROOT_DIR/runtime/third_party/llama.cpp}"
PATCH_FILES=(
  "$ROOT_DIR/runtime/patches/llama_cpp_stage_range.patch"
  "$ROOT_DIR/runtime/patches/llama_cpp_stage_kv_range.patch"
)

if [[ ! -e "$LLAMA_CPP_DIR/.git" ]]; then
  echo "llama.cpp checkout is required at $LLAMA_CPP_DIR" >&2
  exit 2
fi

for patch_file in "${PATCH_FILES[@]}"; do
  patch_name="$(basename "$patch_file")"
  if git -C "$LLAMA_CPP_DIR" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "JetsonFabric llama.cpp patch already applied: $patch_name"
    continue
  fi

  git -C "$LLAMA_CPP_DIR" apply --check "$patch_file"
  git -C "$LLAMA_CPP_DIR" apply "$patch_file"
  echo "Applied JetsonFabric llama.cpp patch: $patch_name"
done
