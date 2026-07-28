#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$WORK_DIR/bin"
cat >"$WORK_DIR/bin/curl" <<'MOCK'
#!/usr/bin/env bash
set -Eeuo pipefail

output_file=""
args=("$@")
for ((index = 0; index < ${#args[@]}; ++index)); do
  if [[ "${args[$index]}" == "-o" ]]; then
    output_file="${args[$((index + 1))]}"
  fi
done

joined=" $* "
if [[ -n "$output_file" ]]; then
  cp "$JF_TEST_RESPONSE" "$output_file"
  printf '200'
elif [[ "$joined" == *"/v1/cluster/members"* ]]; then
  cat "$JF_TEST_MEMBERS"
elif [[ "$joined" == *"/v1/routes/preview"* ]]; then
  cat "$JF_TEST_PREVIEW"
elif [[ "$joined" == *"node-a.test/v1/runtime/deployment"* ]]; then
  cat "$JF_TEST_RUNTIME_A"
elif [[ "$joined" == *"node-b.test/v1/runtime/deployment"* ]]; then
  cat "$JF_TEST_RUNTIME_B"
elif [[ "$joined" == *"node-c.test/v1/runtime/deployment"* ]]; then
  cat "$JF_TEST_RUNTIME_C"
else
  echo "unexpected mock curl request: $*" >&2
  exit 2
fi
MOCK
chmod +x "$WORK_DIR/bin/curl"

MEMBERS="$WORK_DIR/members.json"
PREVIEW="$WORK_DIR/preview.json"
RESPONSE="$WORK_DIR/response.json"
RUNTIME_A="$WORK_DIR/runtime-a.json"
RUNTIME_B="$WORK_DIR/runtime-b.json"
RUNTIME_C="$WORK_DIR/runtime-c.json"

jq -n '{members: [
  {node_id:"node-a",node_name:"node-a",hostname:"host-a",capabilities:{compute_backends:["cuda"],runtime_compute_backend:"cuda",runtime_cuda_active:true,runtime_engine:"llama.cpp",runtime_starts_idle:true,runtime_stage_transport:"http_binary_v1"}},
  {node_id:"node-b",node_name:"node-b",hostname:"host-b",capabilities:{compute_backends:["cuda"],runtime_compute_backend:"cuda",runtime_cuda_active:true,runtime_engine:"llama.cpp",runtime_starts_idle:true,runtime_stage_transport:"http_binary_v1"}},
  {node_id:"node-c",node_name:"node-c",hostname:"host-c",capabilities:{compute_backends:["cuda"],runtime_compute_backend:"cuda",runtime_cuda_active:true,runtime_engine:"llama.cpp",runtime_starts_idle:true,runtime_stage_transport:"http_binary_v1"}}
]}' >"$MEMBERS"

jq -n '{
  valid:true,mode:"pipeline_parallel",topology:"distributed",stage_count:2,physical_host_count:2,
  stages:[
    {stage_index:0,node_id:"node-a",node_name:"node-a",api_url:"http://node-a.test",layer_start:0,layer_end:3},
    {stage_index:1,node_id:"node-b",node_name:"node-b",api_url:"http://node-b.test",layer_start:3,layer_end:6}
  ]
}' >"$PREVIEW"

jq -n '{
  inter_stage_payload_kind:"activation",
  runtime_identity:{engine:"llama.cpp",model_id:"ci-model",model_sha256:"ci-sha",execution_mode:"pipeline_parallel",stage_transport:"http_binary_v1",deployment_id:"ci-deployment",epoch:7},
  plan:{topology:"distributed",physical_host_count:2,stage_count:2,stages:[
    {stage_index:0,node_id:"node-a",node_name:"node-a",api_url:"http://node-a.test",layer_start:0,layer_end:3},
    {stage_index:1,node_id:"node-b",node_name:"node-b",api_url:"http://node-b.test",layer_start:3,layer_end:6}
  ]},
  result:{payload_kind:"sampled_token",sampled_tokens:[11,12],generated_text:"ok",finish_reason:"length",prompt_tokens:4,completion_tokens:2,stages:[
    {phase:"prefill",decode_step:0,stage_index:0,node_name:"node-a",payload_kind_in:"text",payload_kind_out:"activation",payload_in:4,payload_out:32,payload_crc32_in:1,payload_crc32_out:100,latency_ms:1},
    {phase:"prefill",decode_step:0,stage_index:1,node_name:"node-b",payload_kind_in:"activation",payload_kind_out:"sampled_token",payload_in:32,payload_out:4,payload_crc32_in:100,payload_crc32_out:2,latency_ms:1},
    {phase:"decode",decode_step:1,stage_index:0,node_name:"node-a",payload_kind_in:"sampled_token",payload_kind_out:"activation",payload_in:4,payload_out:16,payload_crc32_in:2,payload_crc32_out:200,latency_ms:1},
    {phase:"decode",decode_step:1,stage_index:1,node_name:"node-b",payload_kind_in:"activation",payload_kind_out:"sampled_token",payload_in:16,payload_out:4,payload_crc32_in:200,payload_crc32_out:3,latency_ms:1}
  ]}
}' >"$RESPONSE"

jq -n --argjson layer_start 0 --argjson layer_end 3 '{
  resident:true,active:true,state:"active",
  deployment:{deployment_id:"ci-deployment",epoch:7,model_id:"ci-model",model_sha256:"ci-sha"},
  model_memory:{layer_start:$layer_start,layer_end:$layer_end,layer_count:6,resident_weight_bytes:100,total_weight_bytes:200,resident_tensor_count:10,partitioned:true,pinned:true}
}' >"$RUNTIME_A"
jq -n --argjson layer_start 3 --argjson layer_end 6 '{
  resident:true,active:true,state:"active",
  deployment:{deployment_id:"ci-deployment",epoch:7,model_id:"ci-model",model_sha256:"ci-sha"},
  model_memory:{layer_start:$layer_start,layer_end:$layer_end,layer_count:6,resident_weight_bytes:100,total_weight_bytes:200,resident_tensor_count:10,partitioned:true,pinned:true}
}' >"$RUNTIME_B"
cp "$RUNTIME_B" "$RUNTIME_C"

run_harness() {
  PATH="$WORK_DIR/bin:$PATH" \
  JF_TEST_MEMBERS="$1" \
  JF_TEST_PREVIEW="$2" \
  JF_TEST_RESPONSE="$3" \
  JF_TEST_RUNTIME_A="${4:-$RUNTIME_A}" \
  JF_TEST_RUNTIME_B="${5:-$RUNTIME_B}" \
  JF_TEST_RUNTIME_C="${6:-$RUNTIME_C}" \
  JF_COORDINATOR_URL=http://coordinator.test \
  JF_MODEL_ID=ci-model \
  JF_STAGE_COUNT=2 \
  JF_MAX_TOKENS=2 \
  bash "$ROOT_DIR/scripts/jetson/validate-distributed-cuda.sh" >/dev/null
}

run_harness "$MEMBERS" "$PREVIEW" "$RESPONSE"

INACTIVE_MEMBERS="$WORK_DIR/inactive-members.json"
jq '(.members[] | select(.node_id != "node-a") | .capabilities.runtime_cuda_active) = false' "$MEMBERS" >"$INACTIVE_MEMBERS"
if run_harness "$INACTIVE_MEMBERS" "$PREVIEW" "$RESPONSE" 2>/dev/null; then
  echo "CUDA harness accepted fewer than two CUDA-active hosts" >&2
  exit 1
fi

WRONG_SELECTED_MEMBERS="$WORK_DIR/wrong-selected-members.json"
jq '(.members[] | select(.node_id == "node-b") | .capabilities.runtime_cuda_active) = false' "$MEMBERS" >"$WRONG_SELECTED_MEMBERS"
if run_harness "$WRONG_SELECTED_MEMBERS" "$PREVIEW" "$RESPONSE" 2>/dev/null; then
  echo "CUDA harness accepted a selected runtime without active CUDA" >&2
  exit 1
fi

WRONG_TRANSPORT_MEMBERS="$WORK_DIR/wrong-transport-members.json"
jq '(.members[] | select(.node_id == "node-b") | .capabilities.runtime_stage_transport) = "other"' "$MEMBERS" >"$WRONG_TRANSPORT_MEMBERS"
if run_harness "$WRONG_TRANSPORT_MEMBERS" "$PREVIEW" "$RESPONSE" 2>/dev/null; then
  echo "CUDA harness accepted mismatched stage transports" >&2
  exit 1
fi

WRONG_IDENTITY_RESPONSE="$WORK_DIR/wrong-identity-response.json"
jq '.runtime_identity.model_sha256 = "other-sha"' "$RESPONSE" >"$WRONG_IDENTITY_RESPONSE"
if run_harness "$MEMBERS" "$PREVIEW" "$WRONG_IDENTITY_RESPONSE" 2>/dev/null; then
  echo "CUDA harness accepted a runtime identity that disagrees with live deployments" >&2
  exit 1
fi

WRONG_RUNTIME_B="$WORK_DIR/wrong-runtime-b.json"
jq '.deployment.model_sha256 = "other-sha"' "$RUNTIME_B" >"$WRONG_RUNTIME_B"
if run_harness "$MEMBERS" "$PREVIEW" "$RESPONSE" "$RUNTIME_A" "$WRONG_RUNTIME_B" 2>/dev/null; then
  echo "CUDA harness accepted a live deployment with the wrong model identity" >&2
  exit 1
fi

BAD_CRC_RESPONSE="$WORK_DIR/bad-crc-response.json"
jq '.result.stages[1].payload_crc32_in = 999' "$RESPONSE" >"$BAD_CRC_RESPONSE"
if run_harness "$MEMBERS" "$PREVIEW" "$BAD_CRC_RESPONSE" 2>/dev/null; then
  echo "CUDA harness accepted broken activation CRC continuity" >&2
  exit 1
fi

INACTIVE_RESPONSE_MEMBERS="$WORK_DIR/inactive-response-members.json"
jq '(.members[] | select(.node_id == "node-c") | .capabilities.runtime_cuda_active) = false' "$MEMBERS" >"$INACTIVE_RESPONSE_MEMBERS"
INACTIVE_RESPONSE_PLAN="$WORK_DIR/inactive-response-plan.json"
jq '.plan.stages[1].node_id = "node-c" | .plan.stages[1].node_name = "node-c" | .plan.stages[1].api_url = "http://node-c.test"' "$RESPONSE" >"$INACTIVE_RESPONSE_PLAN"
if run_harness "$INACTIVE_RESPONSE_MEMBERS" "$PREVIEW" "$INACTIVE_RESPONSE_PLAN" 2>/dev/null; then
  echo "CUDA harness accepted an inactive runtime selected by the executed plan" >&2
  exit 1
fi

MISSING_DECODE_RESPONSE="$WORK_DIR/missing-decode-response.json"
jq '.result.sampled_tokens += [13] | .result.completion_tokens = 3' "$RESPONSE" >"$MISSING_DECODE_RESPONSE"
if run_harness "$MEMBERS" "$PREVIEW" "$MISSING_DECODE_RESPONSE" 2>/dev/null; then
  echo "CUDA harness accepted missing later decode traces" >&2
  exit 1
fi

EMPTY_EXECUTED_PLAN="$WORK_DIR/empty-executed-plan.json"
jq '.plan.stages = []' "$RESPONSE" >"$EMPTY_EXECUTED_PLAN"
if run_harness "$MEMBERS" "$PREVIEW" "$EMPTY_EXECUTED_PLAN" 2>/dev/null; then
  echo "CUDA harness accepted an empty executed plan" >&2
  exit 1
fi

echo "Distributed CUDA harness contract tests passed"
