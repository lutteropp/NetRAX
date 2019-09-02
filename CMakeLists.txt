cmake_minimum_required (VERSION 3.0.2 FATAL_ERROR)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")

# The gtest/gmock targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()

# Now simply link your own targets against gtest, gmock,
# etc. as appropriate

set (CMAKE_CXX_FLAGS_DEBUG    "-O3 -g" CACHE INTERNAL "")
set (CMAKE_CXX_FLAGS_RELEASE  "-O3"    CACHE INTERNAL "")
set (CMAKE_C_FLAGS_DEBUG    "-O3 -g" CACHE INTERNAL "")
set (CMAKE_C_FLAGS_RELEASE  "-O3"    CACHE INTERNAL "")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

project (NetRAX C CXX)

message (STATUS "Building ${CMAKE_BUILD_TYPE}")

set (WARN_FLAGS               "-Wall -Wextra")
set (CMAKE_CXX_FLAGS          "-std=c++11 ${WARN_FLAGS}")

message(STATUS "Using flags: ${CMAKE_CXX_FLAGS}")

# build dependencies
set(NETRAX_LOCALDEPS_DIR ${PROJECT_BINARY_DIR}/localdeps)
add_subdirectory(${PROJECT_SOURCE_DIR}/libs)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test/src)

