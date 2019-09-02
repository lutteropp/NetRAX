cmake_minimum_required (VERSION 3.0.2 FATAL_ERROR)

project (NetRAX C CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

set (CMAKE_CXX_FLAGS_DEBUG    "-O3 -g" CACHE INTERNAL "")
set (CMAKE_CXX_FLAGS_RELEASE  "-O3"    CACHE INTERNAL "")
set (CMAKE_C_FLAGS_DEBUG    "-O3 -g" CACHE INTERNAL "")
set (CMAKE_C_FLAGS_RELEASE  "-O3"    CACHE INTERNAL "")

message (STATUS "Building ${CMAKE_BUILD_TYPE}")

set (WARN_FLAGS               "-Wall -Wextra")
set (CMAKE_CXX_FLAGS          "-std=c++11 ${WARN_FLAGS}")

message(STATUS "Using flags: ${CMAKE_CXX_FLAGS}")

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# build dependencies
set(NETRAX_LOCALDEPS_DIR ${PROJECT_BINARY_DIR}/localdeps)
add_subdirectory(${PROJECT_SOURCE_DIR}/libs)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test/src)

