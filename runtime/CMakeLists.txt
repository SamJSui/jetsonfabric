cmake_minimum_required(VERSION 3.16)

project(jetsonfabric_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JF_ENABLE_LLAMA_CPP "Enable llama.cpp engine integration" OFF)
set(JF_LLAMA_CPP_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
  CACHE PATH "Path to a local llama.cpp checkout"
)

add_executable(jetsonfabric-runtime-worker
  worker/main.cpp
  worker/config.cpp
  api/http_response.cpp
  api/http_server.cpp
  protocol/execution_mode.cpp
  protocol/activation.cpp
  pipeline_parallel/stage_assignment.cpp
  pipeline_parallel/stage_worker.cpp
  pipeline_parallel/layer_executor.cpp
  engine/runtime_engine.cpp
)

target_include_directories(jetsonfabric-runtime-worker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(jetsonfabric-runtime-worker PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

if(JF_ENABLE_LLAMA_CPP)
  if(NOT EXISTS "${JF_LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "JF_ENABLE_LLAMA_CPP=ON requires llama.cpp. Run `make setup-llama-cpp` "
      "or set JF_LLAMA_CPP_SOURCE_DIR to a local checkout."
    )
  endif()

  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

  add_subdirectory(
    "${JF_LLAMA_CPP_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/llama.cpp"
    EXCLUDE_FROM_ALL
  )

  target_compile_definitions(jetsonfabric-runtime-worker PRIVATE JF_ENABLE_LLAMA_CPP=1)
  target_link_libraries(jetsonfabric-runtime-worker PRIVATE llama)
else()
  target_compile_definitions(jetsonfabric-runtime-worker PRIVATE JF_ENABLE_LLAMA_CPP=0)
endif()
