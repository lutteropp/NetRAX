cmake_minimum_required (VERSION 3.0.2 FATAL_ERROR)

project (NetRAX C CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

set (CMAKE_CXX_FLAGS_DEBUG    "-O0 -g" CACHE INTERNAL "")
set (CMAKE_CXX_FLAGS_RELEASE  "-O3"    CACHE INTERNAL "")
set (CMAKE_C_FLAGS_DEBUG    "-O0 -g" CACHE INTERNAL "")
set (CMAKE_C_FLAGS_RELEASE  "-O3"    CACHE INTERNAL "")

message (STATUS "Building ${CMAKE_BUILD_TYPE}")

set (WARN_FLAGS               "-Wall -Wextra")
set (CMAKE_CXX_FLAGS          "-std=c++14 ${WARN_FLAGS}")

message(STATUS "Using flags: ${CMAKE_CXX_FLAGS}")

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# include header files
set(NETRAX_LOCALDEPS_DIR ${PROJECT_BINARY_DIR}/localdeps)
set(RAXML_LOCALDEPS_DIR ${PROJECT_BINARY_DIR}/libs/raxml-ng/localdeps)
include_directories(${NETRAX_LOCALDEPS_DIR}/include ${RAXML_LOCALDEPS_DIR}/include ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src)

# build dependencies
add_subdirectory(${PROJECT_SOURCE_DIR}/libs)
# build code
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
# build tests
enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test/src)

