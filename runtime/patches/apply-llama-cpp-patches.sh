#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_CPP_DIR="${1:-$ROOT_DIR/runtime/third_party/llama.cpp}"
PATCH_FILE="$ROOT_DIR/runtime/patches/llama_cpp_stage_range.patch"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  echo "llama.cpp checkout is required at $LLAMA_CPP_DIR" >&2
  exit 2
fi

if git -C "$LLAMA_CPP_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "JetsonFabric llama.cpp stage-range patch already applied"
  exit 0
fi

git -C "$LLAMA_CPP_DIR" apply --check "$PATCH_FILE"
git -C "$LLAMA_CPP_DIR" apply "$PATCH_FILE"
echo "Applied JetsonFabric llama.cpp stage-range patch"
