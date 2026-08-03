if(NOT DEFINED FIXTURE_BIN OR NOT DEFINED COMPILER_BIN OR NOT DEFINED BENCH_BIN OR
   NOT DEFINED INFERENCE_BENCH_BIN OR NOT DEFINED SERVING_TEST_BIN OR
   NOT DEFINED WORK_DIR)
  message(FATAL_ERROR "fixture, compiler, benchmarks, serving test, and work directory are required")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

execute_process(
  COMMAND "${FIXTURE_BIN}" "${WORK_DIR}/fixture.gguf"
  RESULT_VARIABLE FIXTURE_RESULT
  OUTPUT_VARIABLE FIXTURE_OUTPUT
  ERROR_VARIABLE FIXTURE_ERROR
)
if(NOT FIXTURE_RESULT EQUAL 0)
  message(FATAL_ERROR "fixture generation failed: ${FIXTURE_OUTPUT}${FIXTURE_ERROR}")
endif()

execute_process(
  COMMAND "${COMPILER_BIN}"
    --input "${WORK_DIR}/fixture.gguf"
    --output "${WORK_DIR}/fixture.jfm"
  RESULT_VARIABLE COMPILER_RESULT
  OUTPUT_VARIABLE COMPILER_OUTPUT
  ERROR_VARIABLE COMPILER_ERROR
)
if(NOT COMPILER_RESULT EQUAL 0)
  message(FATAL_ERROR "model compilation failed: ${COMPILER_OUTPUT}${COMPILER_ERROR}")
endif()
if(NOT COMPILER_OUTPUT MATCHES "\"layer_count\": 2" OR
   NOT COMPILER_OUTPUT MATCHES "\"tensor_count\": 27")
  message(FATAL_ERROR "model compiler returned unexpected metadata: ${COMPILER_OUTPUT}")
endif()

execute_process(
  COMMAND "${BENCH_BIN}"
    --package "${WORK_DIR}/fixture.jfm"
    --layer-start 0
    --layer-end 1
    --iterations 2
    --prefetch
  RESULT_VARIABLE BENCH_RESULT
  OUTPUT_VARIABLE BENCH_OUTPUT
  ERROR_VARIABLE BENCH_ERROR
)
if(NOT BENCH_RESULT EQUAL 0)
  message(FATAL_ERROR "native model load failed: ${BENCH_OUTPUT}${BENCH_ERROR}")
endif()
if(NOT BENCH_OUTPUT MATCHES "\"selected_tensor_count\": 13" OR
   NOT BENCH_OUTPUT MATCHES "\"total_tensor_count\": 27")
  message(FATAL_ERROR "native model benchmark returned unexpected residency: ${BENCH_OUTPUT}")
endif()

execute_process(
  COMMAND "${INFERENCE_BENCH_BIN}"
    --package "${WORK_DIR}/fixture.jfm"
    --backend cpu
    --tokens 2
    --max-tokens 2
    --iterations 2
    --threads 1
    --expected-tokens 7,7
  RESULT_VARIABLE QWEN_RESULT
  OUTPUT_VARIABLE QWEN_OUTPUT
  ERROR_VARIABLE QWEN_ERROR
)
if(NOT QWEN_RESULT EQUAL 0)
  message(FATAL_ERROR "native Qwen execution failed: ${QWEN_OUTPUT}${QWEN_ERROR}")
endif()
if(NOT QWEN_OUTPUT MATCHES "\"engine\": \"jetsonfabric-native\"" OR
   NOT QWEN_OUTPUT MATCHES "\"sampled_tokens\": \[7,7\]" OR
   NOT QWEN_OUTPUT MATCHES "\"prefill_attention_kernel\": \"unfused\"" OR
   NOT QWEN_OUTPUT MATCHES "\"decode_attention_kernel\": \"unfused\"" OR
   NOT QWEN_OUTPUT MATCHES "\"prefill_plan_reuse_count\": 1" OR
   NOT QWEN_OUTPUT MATCHES "\"prefill_compute_p50_ms\":" OR
   NOT QWEN_OUTPUT MATCHES "\"session_policy\": \"exact_shape_reuse_enabled\"" OR
   NOT QWEN_OUTPUT MATCHES "\"prefill_scratch_bytes\":" OR
   NOT QWEN_OUTPUT MATCHES "\"decode_policy\": \"incremental\"")
  message(FATAL_ERROR "native Qwen benchmark returned unexpected output: ${QWEN_OUTPUT}")
endif()

execute_process(
  COMMAND "${SERVING_TEST_BIN}" "${WORK_DIR}/fixture.jfm"
  RESULT_VARIABLE SERVING_RESULT
  OUTPUT_VARIABLE SERVING_OUTPUT
  ERROR_VARIABLE SERVING_ERROR
)
if(NOT SERVING_RESULT EQUAL 0)
  message(FATAL_ERROR "native serving lifecycle failed: ${SERVING_OUTPUT}${SERVING_ERROR}")
endif()

execute_process(
  COMMAND "${INFERENCE_BENCH_BIN}"
    --package "${WORK_DIR}/fixture.jfm"
    --backend cpu
    --prefill-attention-kernel flash
    --tokens 2
    --max-tokens 1
    --iterations 1
    --threads 1
  RESULT_VARIABLE CPU_FLASH_RESULT
  OUTPUT_VARIABLE CPU_FLASH_OUTPUT
  ERROR_VARIABLE CPU_FLASH_ERROR
)
if(CPU_FLASH_RESULT EQUAL 0 OR
   NOT CPU_FLASH_ERROR MATCHES "flash attention requires the CUDA backend")
  message(FATAL_ERROR
    "native Qwen accepted CUDA flash attention on CPU: "
    "${CPU_FLASH_OUTPUT}${CPU_FLASH_ERROR}"
  )
endif()

execute_process(
  COMMAND "${INFERENCE_BENCH_BIN}"
    --package "${WORK_DIR}/fixture.jfm"
    --backend cpu
    --tokens 2
    --alternate-tokens 2,2
    --max-tokens 2
    --warmups 1
    --iterations 2
    --threads 1
    --session-policy mixed
    --expected-tokens 7,7
    --expected-alternate-tokens 7
  RESULT_VARIABLE MIXED_RESULT
  OUTPUT_VARIABLE MIXED_OUTPUT
  ERROR_VARIABLE MIXED_ERROR
)
if(NOT MIXED_RESULT EQUAL 0 OR
   NOT MIXED_OUTPUT MATCHES "\"sampled_tokens\": \[7,7\]" OR
   NOT MIXED_OUTPUT MATCHES "\"alternate_sampled_tokens\": \[7\]" OR
   NOT MIXED_OUTPUT MATCHES "\"prefill_plan_reuse_count\": 0" OR
   NOT MIXED_OUTPUT MATCHES "\"session_policy\": \"mixed_shape\"")
  message(FATAL_ERROR
    "native mixed-shape execution failed: ${MIXED_OUTPUT}${MIXED_ERROR}"
  )
endif()

execute_process(
  COMMAND "${INFERENCE_BENCH_BIN}"
    --package "${WORK_DIR}/fixture.jfm"
    --backend cpu
    --tokens 2
    --max-tokens 2
    --iterations 2
    --threads 1
    --session-policy cold
    --expected-tokens 7,7
  RESULT_VARIABLE COLD_RESULT
  OUTPUT_VARIABLE COLD_OUTPUT
  ERROR_VARIABLE COLD_ERROR
)
if(NOT COLD_RESULT EQUAL 0 OR
   NOT COLD_OUTPUT MATCHES "\"prefill_plan_reuse_count\": 0" OR
   NOT COLD_OUTPUT MATCHES "\"session_policy\": \"fresh_session\"")
  message(FATAL_ERROR
    "native fresh-session execution failed: ${COLD_OUTPUT}${COLD_ERROR}"
  )
endif()

execute_process(
  COMMAND "${INFERENCE_BENCH_BIN}"
    --package "${WORK_DIR}/fixture.jfm"
    --backend cpu
    --tokens 8
    --max-tokens 1
    --iterations 1
    --threads 1
  RESULT_VARIABLE INVALID_TOKEN_RESULT
  OUTPUT_VARIABLE INVALID_TOKEN_OUTPUT
  ERROR_VARIABLE INVALID_TOKEN_ERROR
)
if(INVALID_TOKEN_RESULT EQUAL 0 OR
   NOT INVALID_TOKEN_ERROR MATCHES "token ID is outside model vocabulary")
  message(FATAL_ERROR
    "native Qwen accepted an invalid token: ${INVALID_TOKEN_OUTPUT}${INVALID_TOKEN_ERROR}"
  )
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
