cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(jetsonfabric_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

set(JF_LLAMA_CPP_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
  CACHE PATH "Path to a local llama.cpp checkout"
)

if(NOT EXISTS "${JF_LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "llama.cpp is required. Run `make setup` or set JF_LLAMA_CPP_SOURCE_DIR."
  )
endif()

if(EXISTS "${JF_LLAMA_CPP_SOURCE_DIR}/.git")
  execute_process(
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/patches/apply-llama-cpp-patches.sh" "${JF_LLAMA_CPP_SOURCE_DIR}"
    RESULT_VARIABLE JF_LLAMA_PATCH_RESULT
    OUTPUT_VARIABLE JF_LLAMA_PATCH_OUTPUT
    ERROR_VARIABLE JF_LLAMA_PATCH_ERROR
  )
  if(NOT JF_LLAMA_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR
      "failed to apply JetsonFabric llama.cpp patch:\n${JF_LLAMA_PATCH_OUTPUT}${JF_LLAMA_PATCH_ERROR}"
    )
  endif()
endif()

file(READ "${JF_LLAMA_CPP_SOURCE_DIR}/src/llama-ext.h" JF_LLAMA_EXT_HEADER)
string(FIND "${JF_LLAMA_EXT_HEADER}" "llama_set_layer_range" JF_LLAMA_STAGE_RANGE_API)
if(JF_LLAMA_STAGE_RANGE_API EQUAL -1)
  message(FATAL_ERROR
    "llama.cpp checkout is missing the JetsonFabric stage-range patch; run runtime/patches/apply-llama-cpp-patches.sh"
  )
endif()

string(FIND "${JF_LLAMA_EXT_HEADER}" "llama_model_resident_tensor_bytes" JF_LLAMA_RESIDENCY_API)
if(JF_LLAMA_RESIDENCY_API EQUAL -1)
  message(FATAL_ERROR
    "llama.cpp checkout is missing the JetsonFabric stage-residency patch; run runtime/patches/apply-llama-cpp-patches.sh"
  )
endif()

file(READ "${JF_LLAMA_CPP_SOURCE_DIR}/src/llama-model.cpp" JF_LLAMA_MODEL_SOURCE)
string(FIND
  "${JF_LLAMA_MODEL_SOURCE}"
  "architecture_filter = std::move(architecture_filter)"
  JF_LLAMA_STAGE_KV_FILTER
)
if(JF_LLAMA_STAGE_KV_FILTER EQUAL -1)
  message(FATAL_ERROR
    "llama.cpp checkout is missing the JetsonFabric stage-local KV patch; run runtime/patches/apply-llama-cpp-patches.sh"
  )
endif()

include(FetchContent)

set(JSON_BuildTests OFF CACHE INTERNAL "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
else()
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  )
endif()
FetchContent_MakeAvailable(nlohmann_json)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_RPC ON CACHE BOOL "" FORCE)

add_subdirectory(
  "${JF_LLAMA_CPP_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/llama.cpp"
  EXCLUDE_FROM_ALL
)

if(TARGET llama)
  set(JF_LLAMA_TARGET llama)
elseif(TARGET llama::llama)
  set(JF_LLAMA_TARGET llama::llama)
else()
  message(FATAL_ERROR "llama.cpp was added, but no llama target was found")
endif()

add_subdirectory(engines/native)

add_executable(jf-model-compile
  compiler/jf_model_compile.cpp
)
target_include_directories(jf-model-compile PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/engines/native/include
  ${CMAKE_CURRENT_SOURCE_DIR}/engines/native/src
  ${JF_LLAMA_CPP_SOURCE_DIR}/ggml/include
)
target_compile_options(jf-model-compile PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(jf-model-compile PRIVATE
  jetsonfabric-native-engine
  ggml
  ${CMAKE_DL_LIBS}
)

if(BUILD_TESTING)
  add_executable(jf-model-fixture
    compiler/model_fixture.cpp
  )
  target_include_directories(jf-model-fixture PRIVATE
    ${JF_LLAMA_CPP_SOURCE_DIR}/ggml/include
  )
  target_compile_options(jf-model-fixture PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jf-model-fixture PRIVATE ggml)
  add_test(
    NAME native-model-compiler
    COMMAND ${CMAKE_COMMAND}
      -DFIXTURE_BIN=$<TARGET_FILE:jf-model-fixture>
      -DCOMPILER_BIN=$<TARGET_FILE:jf-model-compile>
      -DBENCH_BIN=$<TARGET_FILE:jf-native-model-bench>
      -DINFERENCE_BENCH_BIN=$<TARGET_FILE:jf-native-inference-bench>
      -DWORK_DIR=/tmp/jetsonfabric-native-model-compiler-test
      -P ${CMAKE_CURRENT_SOURCE_DIR}/compiler/compiler_integration_test.cmake
  )
endif()

set(JF_RUNTIME_SOURCES
  worker/main.cpp
  worker/config.cpp

  activation/activation_codec_factory.cpp
  activation/builtin_activation_codecs.cpp

  api/http_response.cpp
  api/http_server.cpp

  inference/stage.cpp

  deployment/model_manager.cpp

  protocol/execution_mode.cpp
  protocol/generation.cpp
  protocol/stage.cpp
  protocol/stage_control.cpp
  protocol/utf8.cpp

  adapters/llama_cpp_model.cpp
  adapters/llama_cpp_adapter.cpp
  adapters/llama_cpp_stage_adapter.cpp
  adapters/llama_cpp_tensor_parallel.cpp

  tensor_parallel/device_mesh.cpp

  pipeline_parallel/stage_assignment.cpp
  pipeline_parallel/continuous_batching_executor.cpp
  pipeline_parallel/generation_runner.cpp
  pipeline_parallel/stage_worker.cpp
  pipeline_parallel/llama_cpp_full_model_executor.cpp
  pipeline_parallel/llama_cpp_stage_executor.cpp
  pipeline_parallel/synthetic_activation_executor.cpp

  speculative/prompt_lookup_draft.cpp
  speculative/draft_strategy_factory.cpp

  engine/runtime_service.cpp
  engine/generation_service.cpp
  engine/inference_engine_factory.cpp
  engine/builtin_inference_engines.cpp

  transport/media_type.cpp
  transport/http_stage_transport.cpp
  transport/stage_transport_factory.cpp
  transport/builtin_stage_transports.cpp
)

add_executable(jetsonfabric-runtime-worker
  ${JF_RUNTIME_SOURCES}
)

target_include_directories(jetsonfabric-runtime-worker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${JF_LLAMA_CPP_SOURCE_DIR}/src
)

target_compile_options(jetsonfabric-runtime-worker PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

target_compile_definitions(jetsonfabric-runtime-worker PRIVATE
  JF_ENABLE_LLAMA_CPP=1
)

target_link_libraries(jetsonfabric-runtime-worker PRIVATE
  nlohmann_json::nlohmann_json
  ${JF_LLAMA_TARGET}
)

if(GGML_CUDA)
  add_executable(jetsonfabric-tensor-worker
    tensor_parallel/rpc_worker.cpp
  )
  target_include_directories(jetsonfabric-tensor-worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JF_LLAMA_CPP_SOURCE_DIR}/ggml/include
  )
  target_compile_options(jetsonfabric-tensor-worker PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jetsonfabric-tensor-worker PRIVATE
    ggml
  )
endif()

add_executable(jetsonfabric-two-node-collective-bench
  benchmarks/collective_stats.cpp
  benchmarks/tcp_peer.cpp
  benchmarks/two_node_collective.cpp
)
target_include_directories(jetsonfabric-two-node-collective-bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(jetsonfabric-two-node-collective-bench PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

option(JF_BUILD_CUDA_BENCHMARKS "Build Jetson CUDA feasibility benchmarks" OFF)
set(JF_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures for Jetson benchmarks")
if(JF_BUILD_CUDA_BENCHMARKS)
  if(NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${JF_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(jetsonfabric-tensor-parallel-mlp-bench
    benchmarks/collective_stats.cpp
    benchmarks/tcp_peer.cpp
    benchmarks/tensor_parallel_mlp.cu
  )
  target_include_directories(jetsonfabric-tensor-parallel-mlp-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-tensor-parallel-mlp-bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  target_link_libraries(jetsonfabric-tensor-parallel-mlp-bench PRIVATE
    CUDA::cublas
    CUDA::cudart
  )
  set_target_properties(jetsonfabric-tensor-parallel-mlp-bench PROPERTIES
    CUDA_ARCHITECTURES "${JF_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
endif()

if(BUILD_TESTING)
  add_executable(jetsonfabric-tensor-device-mesh-test
    tensor_parallel/device_mesh.cpp
    tensor_parallel/device_mesh_test.cpp
  )
  target_include_directories(jetsonfabric-tensor-device-mesh-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-tensor-device-mesh-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME tensor-device-mesh
    COMMAND jetsonfabric-tensor-device-mesh-test
  )

  add_executable(jetsonfabric-collective-stats-test
    benchmarks/collective_stats.cpp
    benchmarks/collective_stats_test.cpp
  )
  target_include_directories(jetsonfabric-collective-stats-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-collective-stats-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME collective-stats
    COMMAND jetsonfabric-collective-stats-test
  )

  add_executable(jetsonfabric-runtime-stage-interface-test
    inference/stage.cpp
    inference/stage_test.cpp
  )
  target_include_directories(jetsonfabric-runtime-stage-interface-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-runtime-stage-interface-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME runtime-stage-interface
    COMMAND jetsonfabric-runtime-stage-interface-test
  )

  add_executable(jetsonfabric-model-manager-test
    activation/activation_codec_factory.cpp
    activation/builtin_activation_codecs.cpp
    inference/stage.cpp
    deployment/model_manager.cpp
    engine/inference_engine_factory.cpp
    pipeline_parallel/stage_assignment.cpp
    pipeline_parallel/stage_worker.cpp
    deployment/model_manager_test.cpp
  )
  target_include_directories(jetsonfabric-model-manager-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-model-manager-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME model-manager
    COMMAND jetsonfabric-model-manager-test
  )

  add_executable(jetsonfabric-deployment-lifecycle-test
    activation/activation_codec_factory.cpp
    activation/builtin_activation_codecs.cpp
    inference/stage.cpp
    deployment/model_manager.cpp
    engine/inference_engine_factory.cpp
    pipeline_parallel/stage_assignment.cpp
    pipeline_parallel/stage_worker.cpp
    deployment/lifecycle_test.cpp
  )
  target_include_directories(jetsonfabric-deployment-lifecycle-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-deployment-lifecycle-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME deployment-lifecycle
    COMMAND jetsonfabric-deployment-lifecycle-test
  )

  add_executable(jetsonfabric-generation-runner-test
    inference/stage.cpp
    protocol/generation.cpp
    protocol/stage.cpp
    protocol/stage_control.cpp
    protocol/utf8.cpp
    pipeline_parallel/generation_runner.cpp
    pipeline_parallel/generation_runner_test.cpp
  )
  target_include_directories(jetsonfabric-generation-runner-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-generation-runner-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jetsonfabric-generation-runner-test PRIVATE
    nlohmann_json::nlohmann_json
  )
  add_test(
    NAME generation-runner
    COMMAND jetsonfabric-generation-runner-test
  )

  add_executable(jetsonfabric-http-media-type-test
    transport/media_type.cpp
    transport/media_type_test.cpp
  )
  target_include_directories(jetsonfabric-http-media-type-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-http-media-type-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME http-media-type
    COMMAND jetsonfabric-http-media-type-test
  )

  add_executable(jetsonfabric-kv-cache-type-test
    protocol/kv_cache_type_test.cpp
  )
  target_include_directories(jetsonfabric-kv-cache-type-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-kv-cache-type-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME kv-cache-type
    COMMAND jetsonfabric-kv-cache-type-test
  )

  add_executable(jetsonfabric-http-server-test
    api/http_response.cpp
    api/http_server.cpp
    protocol/execution_mode.cpp
    api/http_server_test.cpp
  )
  target_include_directories(jetsonfabric-http-server-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-http-server-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME http-server-concurrency
    COMMAND jetsonfabric-http-server-test
  )

  add_executable(jetsonfabric-inference-engine-factory-test
    engine/inference_engine_factory.cpp
    engine/inference_engine_factory_test.cpp
  )
  target_include_directories(jetsonfabric-inference-engine-factory-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-inference-engine-factory-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME inference-engine-factory
    COMMAND jetsonfabric-inference-engine-factory-test
  )

  add_executable(jetsonfabric-stage-transport-test
    transport/stage_transport_test.cpp
  )
  target_include_directories(jetsonfabric-stage-transport-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-stage-transport-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME stage-transport
    COMMAND jetsonfabric-stage-transport-test
  )

  add_executable(jetsonfabric-continuous-batching-test
    inference/stage.cpp
    pipeline_parallel/continuous_batching_executor.cpp
    pipeline_parallel/continuous_batching_executor_test.cpp
  )
  target_include_directories(jetsonfabric-continuous-batching-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-continuous-batching-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME continuous-batching
    COMMAND jetsonfabric-continuous-batching-test
  )

  add_executable(jetsonfabric-prompt-lookup-draft-test
    speculative/prompt_lookup_draft.cpp
    speculative/prompt_lookup_draft_test.cpp
  )
  target_include_directories(jetsonfabric-prompt-lookup-draft-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-prompt-lookup-draft-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME prompt-lookup-draft
    COMMAND jetsonfabric-prompt-lookup-draft-test
  )

  add_executable(jetsonfabric-http-stage-transport-test
    protocol/stage.cpp
    protocol/stage_control.cpp
    protocol/utf8.cpp
    transport/media_type.cpp
    transport/http_stage_transport.cpp
    transport/http_stage_transport_test.cpp
  )
  target_include_directories(jetsonfabric-http-stage-transport-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-http-stage-transport-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jetsonfabric-http-stage-transport-test PRIVATE
    nlohmann_json::nlohmann_json
  )
  add_test(
    NAME http-stage-transport
    COMMAND jetsonfabric-http-stage-transport-test
  )

  add_executable(jetsonfabric-stage-transport-factory-test
    transport/stage_transport_factory.cpp
    transport/stage_transport_factory_test.cpp
  )
  target_include_directories(jetsonfabric-stage-transport-factory-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-stage-transport-factory-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME stage-transport-factory
    COMMAND jetsonfabric-stage-transport-factory-test
  )

  add_executable(jetsonfabric-generation-service-test
    activation/activation_codec_factory.cpp
    activation/builtin_activation_codecs.cpp
    inference/stage.cpp
    deployment/model_manager.cpp
    engine/inference_engine_factory.cpp
    pipeline_parallel/stage_assignment.cpp
    pipeline_parallel/stage_worker.cpp
    pipeline_parallel/generation_runner.cpp
    protocol/utf8.cpp
    engine/generation_service.cpp
    engine/generation_service_test.cpp
  )
  target_include_directories(jetsonfabric-generation-service-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-generation-service-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME generation-service
    COMMAND jetsonfabric-generation-service-test
  )

  add_executable(jetsonfabric-activation-codec-test
    activation/activation_codec_factory.cpp
    activation/builtin_activation_codecs.cpp
    inference/stage.cpp
    activation/activation_codec_test.cpp
  )
  target_include_directories(jetsonfabric-activation-codec-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-activation-codec-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME activation-codec
    COMMAND jetsonfabric-activation-codec-test
  )

  add_executable(jetsonfabric-stage-worker-codec-test
    activation/activation_codec_factory.cpp
    activation/builtin_activation_codecs.cpp
    inference/stage.cpp
    pipeline_parallel/stage_assignment.cpp
    pipeline_parallel/stage_worker.cpp
    pipeline_parallel/stage_worker_codec_test.cpp
  )
  target_include_directories(jetsonfabric-stage-worker-codec-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(jetsonfabric-stage-worker-codec-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  add_test(
    NAME stage-worker-codec
    COMMAND jetsonfabric-stage-worker-codec-test
  )

  add_test(
    NAME runtime-rejects-unknown-engine
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--engine
      -DVALUE=missing-engine
      "-DEXPECTED=unsupported inference engine: missing-engine"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-unknown-stage-transport
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--stage-transport
      -DVALUE=missing-transport
      "-DEXPECTED=unsupported stage transport: missing-transport"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-unknown-activation-encoding
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--activation-encoding
      -DVALUE=missing-encoding
      "-DEXPECTED=unsupported activation encoding: missing-encoding"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-unknown-kv-cache-type
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--kv-cache-type
      -DVALUE=missing-kv-cache-type
      "-DEXPECTED=unknown KV cache type: missing-kv-cache-type"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-invalid-ubatch-size
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--ubatch-size
      -DVALUE=0
      "-DEXPECTED=--ubatch-size must be greater than zero"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-unknown-speculative-draft
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--speculative-draft
      -DVALUE=missing-draft
      "-DEXPECTED=--speculative-draft must be none or prompt_lookup"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-tensor-without-rpc-peer
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--mode
      -DVALUE=tensor_parallel
      "-DEXPECTED=tensor_parallel requires at least one remote RPC endpoint"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-tensor-peer-outside-tensor-mode
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--tensor-rpc-peers
      -DVALUE=127.0.0.1:52520
      "-DEXPECTED=tensor RPC peers and tensor split require --mode tensor_parallel"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-tensor-synthetic-engine
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--mode
      -DVALUE=tensor_parallel
      "-DEXTRA_ARGS=--tensor-rpc-peers;127.0.0.1:52520;--engine;synthetic"
      "-DEXPECTED=tensor_parallel currently requires --engine llama.cpp"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-quantized-tensor-kv-cache
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--mode
      -DVALUE=tensor_parallel
      "-DEXTRA_ARGS=--tensor-rpc-peers;127.0.0.1:52520;--kv-cache-type;q8_0"
      "-DEXPECTED=tensor_parallel currently requires --kv-cache-type f16"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_test(
    NAME runtime-rejects-unauthenticated-external-listen
    COMMAND ${CMAKE_COMMAND}
      -DRUNTIME_BIN=$<TARGET_FILE:jetsonfabric-runtime-worker>
      -DOPTION=--listen
      -DVALUE=0.0.0.0:9090
      "-DEXPECTED=non-loopback --listen requires JETSONFABRIC_CLUSTER_TOKEN"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/worker/startup_rejection_test.cmake
  )

  add_executable(jetsonfabric-llama-stage-test
    inference/stage.cpp
    tensor_parallel/device_mesh.cpp
    adapters/llama_cpp_model.cpp
    adapters/llama_cpp_adapter.cpp
    adapters/llama_cpp_stage_adapter.cpp
    adapters/llama_cpp_tensor_parallel.cpp
    adapters/llama_cpp_stage_adapter_test.cpp
  )
  target_include_directories(jetsonfabric-llama-stage-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JF_LLAMA_CPP_SOURCE_DIR}/src
  )
  target_compile_options(jetsonfabric-llama-stage-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jetsonfabric-llama-stage-test PRIVATE
    ${JF_LLAMA_TARGET}
  )
  add_test(
    NAME llama-stage-equivalence
    COMMAND jetsonfabric-llama-stage-test
  )
endif()
