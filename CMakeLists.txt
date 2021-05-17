cmake_minimum_required (VERSION 3.0.2 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)

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

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(NETRAX_BINARY_NAME "" CACHE STRING "Custom NetRAX binary name")
set(USE_LIBPLL_CMAKE ON CACHE BOOL "Use CMake to build libpll and pll-modules")
set(BUILD_AS_LIBRARY ON CACHE BOOL "Build NetRAX as shared library (instead of stand-alone executable)")
set(PLLMOD_DEBUG OFF CACHE BOOL "Debug pll-modules (tons of console output!)")

set (USE_PTHREADS ON CACHE BOOL "Enable multi-threading support (PTHREADS)")
set (USE_MPI ON CACHE BOOL "Enable MPI support")
set (USE_VCF OFF)

# set both following options to OFF to build a portable binary 
# (don't worry, libpll will still have full SIMD support!)
set (ENABLE_RAXML_SIMD OFF CACHE BOOL "Enable SIMD instructions in RAxML (non-portable but slightly faster)")
set (ENABLE_PLLMOD_SIMD OFF CACHE BOOL "Enable SIMD instructions in pll-modules (non-portable but slightly faster)")

# build a static binary
set(STATIC_BUILD OFF CACHE BOOL "Build static binary")

set(USE_TERRAPHAST ON CACHE BOOL "Use phylogentic terraces library (terraphast) ")
set(USE_GMP OFF CACHE BOOL "Use GNU Multiple precision (GMP) library ")

set(ENABLE_PTHREADS ${USE_PTHREADS})

if(USE_TERRAPHAST)
  set(GCC_MIN_VERSION "5.4")
  set(CLANG_MIN_VERSION "3.8")
  set(ENABLE_GMP ${USE_GMP})
else()
  set(GCC_MIN_VERSION "4.8")
  set(CLANG_MIN_VERSION "3.8")
  set(ENABLE_GMP OFF)
endif()

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_EXTENSIONS OFF)

# set these flags globally for all subprojects (libpll etc.)
set (CMAKE_CXX_FLAGS_DEBUG    "-O2 -g" CACHE INTERNAL "")
set (CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG"    CACHE INTERNAL "")
set (CMAKE_C_FLAGS_DEBUG    "-O2 -g" CACHE INTERNAL "")
set (CMAKE_C_FLAGS_RELEASE  "-O3 -DNDEBUG"    CACHE INTERNAL "")

project (netrax C CXX)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # no support for static linking on Mac
  set(STATIC_BUILD OFF)
  # MPI support on Macs seems to be tricky (and unnecessary)
  set(ENABLE_MPI OFF)
else()
  set(ENABLE_MPI ${USE_MPI})
endif()

#check for minimum compiler versions
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} => ${CMAKE_CXX_COMPILER}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(RAXML_COMPILER_TARGET_VERSION ${GCC_MIN_VERSION})
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(RAXML_COMPILER_TARGET_VERSION ${CLANG_MIN_VERSION})
endif()


if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS RAXML_COMPILER_TARGET_VERSION)
  message (FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} compiler too old! Minimum required: ${RAXML_COMPILER_TARGET_VERSION}")
endif()


set (netrax_VERSION_MAJOR 1)
set (netrax_VERSION_MINOR 0)

#set (CMAKE_BUILD_TYPE DEBUG)
#set (CMAKE_BUILD_TYPE RELEASE)
# set (CMAKE_VERBOSE_MAKEFILE ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

message (STATUS "Building ${CMAKE_BUILD_TYPE}")

set (WARN_FLAGS               "-Wall -Wextra")
set (CMAKE_CXX_FLAGS          "-std=c++14 ${WARN_FLAGS}")

if (ENABLE_GMP)
#find_package(GMP REQUIRED)
  set(GMP_FOUND ON)
  set(GMP_LIBRARIES gmpxx gmp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_RAXML_GMP")
endif()

if (ENABLE_RAXML_SIMD)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG(-mavx HAS_AVX)
  CHECK_CXX_COMPILER_FLAG(-msse3 HAS_SSE3)
  if(HAS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -D__AVX")
  elseif(HAS_SSE3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -D__SSE3")
  endif()
endif()

if(ENABLE_PLLMOD_SIMD)
  set(PLLMOD_AUTOCONF_ARGS "--enable-avx --enable-sse")
else()
  set(PLLMOD_AUTOCONF_ARGS "")
endif()

if(ENABLE_PTHREADS)
  find_package(Threads REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_RAXML_PTHREADS -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_RAXML_PTHREADS -pthread")
endif()


if(BUILD_AS_LIBRARY)
  # building shared library only works with CMake!
  set(USE_LIBPLL_CMAKE ON)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_RAXML_BUILD_AS_LIB -fPIC")
endif()

if(USE_TERRAPHAST)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_RAXML_TERRAPHAST")
endif()

message(STATUS "Using flags: ${CMAKE_CXX_FLAGS}")

set(NETRAX_LOCALDEPS_DIR ${PROJECT_BINARY_DIR}/localdeps)
set(RAXML_LOCALDEPS_DIR ${PROJECT_BINARY_DIR}/libs/raxml-ng/localdeps)
include_directories(${RAXML_LOCALDEPS_DIR}/include ${NETRAX_LOCALDEPS_DIR}/include ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src ${TERRACES_INCLUDE_DIR})

# build dependencies
add_subdirectory(${PROJECT_SOURCE_DIR}/libs)

find_package(MPI)
if(MPI_CXX_FOUND)
  # set( ENV{OMPI_CXX}            "clang++" PARENT_SCOPE )
  set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILER_FLAGS} -D_RAXML_MPI")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")

  # disable static build for MPI
  set(STATIC_BUILD OFF)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test/src)