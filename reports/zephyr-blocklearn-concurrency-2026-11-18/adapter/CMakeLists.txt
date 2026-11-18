cmake_minimum_required(VERSION 3.16)
project(zephyr_concurrent_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ZEPHYR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../_deps/zephyr/Learned-Index-Model-main/src"
    CACHE PATH "Zephyr Learned-Index-Model src directory")

if(NOT EXISTS "${ZEPHYR_SRC}/CMakeLists.txt")
  message(FATAL_ERROR "ZEPHYR_SRC is not the Zephyr src directory: ${ZEPHYR_SRC}")
endif()

# Reuse the upstream BlockLearn implementation and its v1 factories.
add_subdirectory("${ZEPHYR_SRC}" "${CMAKE_CURRENT_BINARY_DIR}/zephyr-build")

add_library(zephyr_concurrent_adapter STATIC zephyr_concurrent_adapter.cpp)
target_compile_features(zephyr_concurrent_adapter PUBLIC cxx_std_17)
target_include_directories(zephyr_concurrent_adapter PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}")
if(DEFINED STRING_INDEX_CONTRACT_INCLUDE_DIR)
  target_include_directories(zephyr_concurrent_adapter PUBLIC
    "${STRING_INDEX_CONTRACT_INCLUDE_DIR}")
else()
  target_include_directories(zephyr_concurrent_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../learned-index/src")
endif()
target_link_libraries(zephyr_concurrent_adapter PUBLIC blocklearn_adapter)
