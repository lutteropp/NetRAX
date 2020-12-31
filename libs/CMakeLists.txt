
message(STATUS "Building dependencies in: ${NETRAX_LOCALDEPS_DIR}")

set(BUILD_AS_LIBRARY)

add_subdirectory(raxml-ng)

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

set(RAXML_LIBS ${RAXML_LIBS} PARENT_SCOPE)



