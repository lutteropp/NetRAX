
message (STATUS "Building tests")

file (GLOB_RECURSE NETRAX_TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/src/*.cpp ${NETRAX_SOURCES})

# sources list now has 2 Main.cpp, old has to be removed
list(REMOVE_ITEM NETRAX_TEST_SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")

include_directories (${PROJECT_SOURCE_DIR})

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test/bin)

add_executable        (netrax_test_module ${NETRAX_TEST_SOURCES})

target_link_libraries(netrax_test_module "${PROJECT_BINARY_DIR}/../libs/raxml-ng/bin/raxml-ng-mpi.a" ${NETRAX_LIBS})

target_link_libraries (netrax_test_module gtest_main -pthread)

target_link_libraries (netrax_test_module ${MPI_CXX_LIBRARIES})

set_target_properties (netrax_test_module PROPERTIES OUTPUT_NAME netrax_test)
set_target_properties (netrax_test_module PROPERTIES PREFIX "")

add_test (netrax_test ${PROJECT_SOURCE_DIR}/test/bin/netrax_test)
set_tests_properties(netrax_test PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)
