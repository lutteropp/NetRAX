file(GLOB NETRAX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/io/*.c
                         ${PROJECT_SOURCE_DIR}/src/io/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/likelihood/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/optimization/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/traversal/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/graph/*.cpp
                        )

# export to the parent dir to use in GTest makefile
set(NETRAX_SOURCES ${NETRAX_SOURCES} PARENT_SCOPE)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories("../libs/" ${CMAKE_CURRENT_SOURCE_DIR}/io)

# export to the parent dir to use in GTest makefile
set(NETRAX_SOURCES ${NETRAX_SOURCES} PARENT_SCOPE)
set(NETRAX_LIBS pthread "${PROJECT_BINARY_DIR}/../libs/raxml-ng/bin/raxml-ng.a" ${RAXML_LIBS} gmp mpfr)
set(NETRAX_LIBS ${NETRAX_LIBS} PARENT_SCOPE)

add_executable(netrax ${NETRAX_SOURCES})

#message(STATUS "Using NETRAX_LIBS: ${NETRAX_LIBS}")

target_link_libraries(netrax ${NETRAX_LIBS})

