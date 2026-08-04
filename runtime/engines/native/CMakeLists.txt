cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(jetsonfabric_native_engine LANGUAGES C CXX)
  include(CTest)
endif()

find_package(Threads REQUIRED)

add_library(jetsonfabric-native-engine STATIC
  src/engine.c
  src/sha256.c
)
target_include_directories(jetsonfabric-native-engine
  PUBLIC include
  PRIVATE src
)
target_compile_features(jetsonfabric-native-engine PUBLIC c_std_11)
target_compile_options(jetsonfabric-native-engine PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(jetsonfabric-native-engine PUBLIC Threads::Threads)

add_executable(jf-native-model-bench
  benchmarks/model_load_bench.cpp
)
target_compile_features(jf-native-model-bench PRIVATE cxx_std_20)
target_compile_options(jf-native-model-bench PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(jf-native-model-bench PRIVATE
  jetsonfabric-native-engine
)

if(TARGET ggml)
  add_library(jetsonfabric-native-inference STATIC
    src/architecture_registry.cpp
    src/native_inference.cpp
    src/qwen2_architecture.cpp
    src/qwen2_metadata.cpp
    src/stage_memory_estimator.cpp
    src/tensor_store.cpp
  )
  target_include_directories(jetsonfabric-native-inference
    PUBLIC include
    PRIVATE src
  )
  target_compile_features(jetsonfabric-native-inference PUBLIC cxx_std_20)
  target_compile_options(jetsonfabric-native-inference PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jetsonfabric-native-inference
    PUBLIC jetsonfabric-native-engine
    PRIVATE ggml
  )
  add_executable(jf-native-inference-bench
    benchmarks/native_inference_bench.cpp
  )
  target_compile_features(jf-native-inference-bench PRIVATE cxx_std_20)
  target_compile_options(jf-native-inference-bench PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jf-native-inference-bench PRIVATE
    jetsonfabric-native-inference
  )
  add_executable(jf-native-attention-parity
    benchmarks/native_attention_parity.cpp
  )
  target_compile_features(jf-native-attention-parity PRIVATE cxx_std_20)
  target_compile_options(jf-native-attention-parity PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_link_libraries(jf-native-attention-parity PRIVATE
    jetsonfabric-native-inference
  )
endif()

if(BUILD_TESTING)
  add_executable(jetsonfabric-native-engine-test
    tests/engine_test.cpp
  )
  target_compile_features(jetsonfabric-native-engine-test PRIVATE cxx_std_20)
  target_compile_options(jetsonfabric-native-engine-test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_include_directories(jetsonfabric-native-engine-test PRIVATE src)
  target_compile_definitions(jetsonfabric-native-engine-test PRIVATE
    JF_NATIVE_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
  target_link_libraries(jetsonfabric-native-engine-test PRIVATE
    jetsonfabric-native-engine
  )
  add_test(NAME native-engine COMMAND jetsonfabric-native-engine-test)
endif()
