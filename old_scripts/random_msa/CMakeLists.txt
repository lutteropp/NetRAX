message (STATUS "Building scripts")

file (GLOB_RECURSE RANDOM_MSA_SOURCES ${PROJECT_SOURCE_DIR}/scripts/generate_random_msa.cpp ${NETRAX_SOURCES})

# sources list now has 2 Main.cpp, old has to be removed
list(REMOVE_ITEM RANDOM_MSA_SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")

include_directories (${PROJECT_SOURCE_DIR})

include_directories("../libs/" $${PROJECT_SOURCE_DIR}/src/io)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/scripts/bin)

add_executable        (random_msa_module ${RANDOM_MSA_SOURCES})

target_link_libraries(random_msa_module "${PROJECT_BINARY_DIR}/../libs/raxml-ng/bin/raxml-ng.a" ${NETRAX_LIBS})

target_link_libraries(random_msa_module ${NETRAX_LIBS})

target_link_libraries (random_msa_module ${MPI_CXX_LIBRARIES})

set_target_properties (random_msa_module PROPERTIES OUTPUT_NAME random_msa)
set_target_properties (random_msa_module PROPERTIES PREFIX "")
