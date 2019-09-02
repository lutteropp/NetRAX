find_package (GTest)

if(NOT GTEST_FOUND)
    message (STATUS "GTest not found")
    message (WARNING "Skipping building tests.")
    return()
endif()

message (STATUS "Building tests")

file (GLOB_RECURSE NETRAX_TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/src/*.cpp ${SRC_FILES})

# sources list now has 2 Main.cpp, old has to be removed
list(REMOVE_ITEM NETRAX_TEST_SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")

include_directories (${PROJECT_SOURCE_DIR})

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test/bin)

add_executable        (netrax_test_module ${NETRAX_TEST_SOURCES})

target_link_libraries (netrax_test_module ${NETRAX_LIBS})

target_link_libraries (netrax_test_module ${GTEST_BOTH_LIBRARIES} -pthread)

target_link_libraries (netrax_test_module ${MPI_CXX_LIBRARIES})

set_target_properties (netrax_test_module PROPERTIES OUTPUT_NAME netrax_test)
set_target_properties (netrax_test_module PROPERTIES PREFIX "")


add_test (netrax_test ${PROJECT_SOURCE_DIR}/test/bin/netrax_test)
