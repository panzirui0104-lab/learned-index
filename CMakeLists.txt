cmake_minimum_required(VERSION 3.16)
project(context16_concurrent_index LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(context16_concurrent_index INTERFACE)
target_include_directories(context16_concurrent_index INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/concurrent")

add_executable(concurrent_index_test tests/concurrent_index_test.cpp)
target_link_libraries(concurrent_index_test PRIVATE context16_concurrent_index)

find_package(Threads QUIET)
if(TARGET Threads::Threads)
  target_link_libraries(concurrent_index_test PRIVATE Threads::Threads)
endif()

enable_testing()
add_test(NAME concurrent_index_test COMMAND concurrent_index_test)
