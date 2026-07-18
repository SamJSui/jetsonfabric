#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${CI_MODEL_PATH:?CI_MODEL_PATH is required}" \
MODEL_ID="ci-tiny-llama" \
RUNTIME_BUILD_DIR="${CI_RUNTIME_BUILD_DIR:-$ROOT_DIR/runtime/build-ci-cpu}" \
RUNTIME_BIN="$ROOT_DIR/dist/jetsonfabric-runtime-worker-ci-cpu" \
NODE_BIN="$ROOT_DIR/dist/jetsonfabric-node" \
RUNTIME_BUILD_JOBS="${CI_RUNTIME_BUILD_JOBS:-2}" \
JF_CTX_SIZE="${CI_SINGLE_CTX_SIZE:-128}" \
JF_MAX_TOKENS="${CI_SINGLE_MAX_TOKENS:-2}" \
JF_RAW_PROMPT="Once upon a time" \
JF_CHAT_PROMPT="Hi" \
bash "$ROOT_DIR/scripts/local/validate-single-node.sh"
