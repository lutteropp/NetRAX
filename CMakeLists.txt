cmake_minimum_required (VERSION 3.0.2 FATAL_ERROR)

project (NetRAX C CXX)


# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windowss
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

set (CMAKE_CXX_FLAGS_DEBUG    "-O0 -g" CACHE INTERNAL "")
set (CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG"    CACHE INTERNAL "")
set (CMAKE_C_FLAGS_DEBUG    "-O0 -g" CACHE INTERNAL "")
set (CMAKE_C_FLAGS_RELEASE  "-O3 -DNDEBUG"    CACHE INTERNAL "")

set(CMAKE_CXX_STANDARD 17)

message (STATUS "Building ${CMAKE_BUILD_TYPE}")

set (WARN_FLAGS               "-Wall -Wextra")
set (CMAKE_CXX_FLAGS          "-std=c++17 ${WARN_FLAGS}")

message(STATUS "Using flags: ${CMAKE_CXX_FLAGS}")

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
# build scripts
add_subdirectory(${PROJECT_SOURCE_DIR}/scripts)

