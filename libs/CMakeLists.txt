
message(STATUS "Building dependencies in: ${NETRAX_LOCALDEPS_DIR}")

set(BUILD_AS_LIBRARY)

add_subdirectory(raxml-ng)

file(COPY ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/PartitionedMSA.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/MSA.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/constants.hpp
            DESTINATION ${NETRAX_LOCALDEPS_DIR}/include/raxml-ng)



