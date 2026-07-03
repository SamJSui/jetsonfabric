cmake_minimum_required(VERSION 3.16)

project(jetsonfabric_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(jetsonfabric-runtime-worker
  worker/main.cpp
  worker/config.cpp
  api/http_response.cpp
  api/http_server.cpp
  engine/stub_engine.cpp
)

target_include_directories(jetsonfabric-runtime-worker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(jetsonfabric-runtime-worker PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
