file(GLOB NETRAX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp
                        ${PROJECT_SOURCE_DIR}/src/io/*.cpp
                        )

# export to the parent dir to use in GTest makefile
set(NETRAX_SOURCES ${NETRAX_SOURCES} PARENT_SCOPE)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set_target_properties(netrax_module PROPERTIES PREFIX "")

install(TARGETS netrax_module DESTINATION bin)
