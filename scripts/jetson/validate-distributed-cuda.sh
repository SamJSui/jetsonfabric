#!/usr/bin/env bash
set -Eeuo pipefail

COORDINATOR_URL="${JF_COORDINATOR_URL:?JF_COORDINATOR_URL is required, for example http://jetson-a:8080}"
MODEL_ID="${JF_MODEL_ID:?JF_MODEL_ID is required}"
STAGE_COUNT="${JF_STAGE_COUNT:-2}"
MAX_TOKENS="${JF_MAX_TOKENS:-2}"
PROMPT="${JF_PROMPT:-Once upon a time}"
EXPECTED_TOKENS="${JF_EXPECTED_TOKENS:-}"
ALLOW_COLOCATED="${JF_ALLOW_COLOCATED_STAGES:-false}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}
require_command curl
require_command jq

api() {
  printf '%s/%s' "${COORDINATOR_URL%/}" "${1#/}"
}

members_json="$(curl -fsS "$(api /v1/cluster/members)")"

cuda_hosts="$(jq -r '
  [.members[]
    | select(
        ((.capabilities.compute_backends // []) | index("cuda")) and
        .capabilities.runtime_compute_backend == "cuda" and
        .capabilities.runtime_cuda_active == true)
    | (.hostname // .node_name)]
  | unique
  | length
' <<<"$members_json")"
if (( cuda_hosts < 2 )); then
  echo "expected at least two distinct CUDA-active physical hosts, found $cuda_hosts" >&2
  jq '.members | map({
    node_name,
    hostname,
    compute_backends: .capabilities.compute_backends,
    runtime_compute_backend: .capabilities.runtime_compute_backend,
    runtime_cuda_active: .capabilities.runtime_cuda_active
  })' <<<"$members_json" >&2
  exit 1
fi

preview_url="$(api /v1/routes/preview)?model=$(printf '%s' "$MODEL_ID" | jq -sRr @uri)&stage_count=$STAGE_COUNT&allow_colocated_stages=$ALLOW_COLOCATED"
preview_json="$(curl -fsS "$preview_url")"
jq -e --argjson count "$STAGE_COUNT" '
  .valid == true and
  .mode == "pipeline_parallel" and
  .topology == "distributed" and
  .stage_count == $count and
  .physical_host_count >= 2 and
  (.stages | length) == $count
' <<<"$preview_json" >/dev/null

jq -e --argjson preview "$preview_json" '
  .members as $members |
  all($preview.stages[];
    .node_id as $node_id |
    any($members[];
      .node_id == $node_id and
      ((.capabilities.compute_backends // []) | index("cuda")) and
      .capabilities.runtime_compute_backend == "cuda" and
      .capabilities.runtime_cuda_active == true))
' <<<"$members_json" >/dev/null

request_file="$(mktemp)"
response_file="$(mktemp)"
cleanup() {
  rm -f "$request_file" "$response_file"
}
trap cleanup EXIT

jq -n \
  --arg model "$MODEL_ID" \
  --arg payload "$PROMPT" \
  --argjson max_tokens "$MAX_TOKENS" \
  --argjson stage_count "$STAGE_COUNT" \
  --argjson allow_colocated "$ALLOW_COLOCATED" \
  '{
    request_id: ("physical-jetson-" + (now | tostring)),
    model: $model,
    payload: $payload,
    max_tokens: $max_tokens,
    stage_count: $stage_count,
    allow_colocated_stages: $allow_colocated
  }' >"$request_file"

http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
  -X POST "$(api /v1/layer-split/run)" \
  -H 'Content-Type: application/json' \
  --data-binary "@$request_file")"
if [[ "$http_code" != "200" ]]; then
  echo "distributed CUDA request failed with HTTP $http_code" >&2
  cat "$response_file" >&2
  exit 1
fi

jq -e --argjson count "$STAGE_COUNT" '
  .inter_stage_payload_kind == "activation" and
  .plan.topology == "distributed" and
  .plan.physical_host_count >= 2 and
  .plan.stage_count == $count and
  .result.payload_kind == "sampled_token" and
  (.result.sampled_tokens | length) >= 1 and
  .result.finish_reason != "" and
  ([.result.stages[] | select(.phase == "prefill")] | length) == $count and
  ([.result.stages[] | select(.phase == "prefill" and .payload_kind_out == "activation")] | length) >= 1 and
  ([.result.stages[] | select(.payload_kind_in == "activation")] | length) >= 1 and
  .result.stages as $traces |
  all($traces[];
    if .payload_kind_out == "activation" then
      . as $source |
      any($traces[];
        .phase == $source.phase and
        .decode_step == $source.decode_step and
        .stage_index == ($source.stage_index + 1) and
        .payload_kind_in == "activation" and
        .payload_in == $source.payload_out and
        .payload_crc32_in == $source.payload_crc32_out)
    else true end)
' "$response_file" >/dev/null

if (( MAX_TOKENS > 1 )); then
  generated_count="$(jq '.result.sampled_tokens | length' "$response_file")"
  if (( generated_count > 1 )); then
    jq -e --argjson count "$STAGE_COUNT" '
      ([.result.stages[] | select(.phase == "decode" and .decode_step == 1)] | length) == $count
    ' "$response_file" >/dev/null
  fi
fi

if [[ -n "$EXPECTED_TOKENS" ]]; then
  jq -e --argjson expected "$EXPECTED_TOKENS" '.result.sampled_tokens == $expected' "$response_file" >/dev/null || {
    echo "distributed sampled tokens do not match JF_EXPECTED_TOKENS" >&2
    jq '{expected: $expected, actual: .result.sampled_tokens}' --argjson expected "$EXPECTED_TOKENS" "$response_file" >&2
    exit 1
  }
fi

echo "Physical distributed CUDA validation passed"
jq '{
  topology: .plan.topology,
  physical_host_count: .plan.physical_host_count,
  stages: [.plan.stages[] | {stage_index, node_name, layer_start, layer_end}],
  sampled_tokens: .result.sampled_tokens,
  generated_text: .result.generated_text,
  finish_reason: .result.finish_reason,
  prompt_tokens: .result.prompt_tokens,
  completion_tokens: .result.completion_tokens,
  traces: [.result.stages[] | {
    phase, decode_step, stage_index, node_name,
    payload_kind_in, payload_kind_out,
    payload_in, payload_out,
    payload_crc32_in, payload_crc32_out,
    latency_ms
  }]
}' "$response_file"
