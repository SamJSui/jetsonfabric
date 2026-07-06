cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(jetsonfabric_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(JF_LLAMA_CPP_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
  CACHE PATH "Path to a local llama.cpp checkout"
)

if(NOT EXISTS "${JF_LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "llama.cpp is required. Run `make setup` or set JF_LLAMA_CPP_SOURCE_DIR."
  )
endif()

include(FetchContent)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(
  "${JF_LLAMA_CPP_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/llama.cpp"
  EXCLUDE_FROM_ALL
)

set(JF_RUNTIME_SOURCES
  worker/main.cpp
  worker/config.cpp

  api/http_response.cpp
  api/http_server.cpp

  protocol/execution_mode.cpp
  protocol/activation.cpp

  adapters/llama_cpp_adapter.cpp

  pipeline_parallel/stage_assignment.cpp
  pipeline_parallel/stage_worker.cpp
  pipeline_parallel/llama_cpp_full_model_executor.cpp

  engine/runtime_engine.cpp
  engine/runtime_engine_factory.cpp
)

add_executable(jetsonfabric-runtime-worker
  ${JF_RUNTIME_SOURCES}
)

target_include_directories(jetsonfabric-runtime-worker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
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
)

if(TARGET llama)
  target_link_libraries(jetsonfabric-runtime-worker PRIVATE llama)
elseif(TARGET llama::llama)
  target_link_libraries(jetsonfabric-runtime-worker PRIVATE llama::llama)
else()
  message(FATAL_ERROR "llama.cpp was added, but no llama target was found")
endif()