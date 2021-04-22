
message(STATUS "Building dependencies in: ${NETRAX_LOCALDEPS_DIR}")

set(BUILD_AS_LIBRARY)

add_subdirectory(raxml-ng)

set(NETRAX_LOCALDEPS_DIR ${NETRAX_LOCALDEPS_DIR} PARENT_SCOPE)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

file(COPY ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/PartitionedMSA.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/MSA.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/constants.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/common.h
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/TreeInfo.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/Model.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/Optimizer.hpp
	        ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/main.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/src/log.hpp
            DESTINATION ${NETRAX_LOCALDEPS_DIR}/include/raxml-ng)

file(COPY ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/advanced.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/bigint.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/definitions.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/clamped_uint.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/bitmatrix.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/trees.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/constraints.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/parser.hpp
          ${PROJECT_SOURCE_DIR}/libs/raxml-ng/libs/terraphast/include/terraces/errors.hpp
          DESTINATION ${NETRAX_LOCALDEPS_DIR}/include/terraces)


set(RAXML_LIBS ${RAXML_LIBS} PARENT_SCOPE)



