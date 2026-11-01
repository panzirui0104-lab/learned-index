cmake_minimum_required(VERSION 3.16)
project(context16_olc_index LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(olc_context16 INTERFACE)
target_include_directories(olc_context16 INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads QUIET)
enable_testing()

foreach(test_name olc_index_test olc_random_test olc_value_test olc_contention_test)
  add_executable(${test_name} tests/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE olc_context16)
  if(TARGET Threads::Threads)
    target_link_libraries(${test_name} PRIVATE Threads::Threads)
  endif()
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
