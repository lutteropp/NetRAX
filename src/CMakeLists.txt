file(GLOB NETRAX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/io/*.c
                         ${PROJECT_SOURCE_DIR}/src/io/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/likelihood/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/optimization/*.cpp
                         ${PROJECT_SOURCE_DIR}/src/traversal/*.cpp
                        )

# export to the parent dir to use in GTest makefile
set(NETRAX_SOURCES ${NETRAX_SOURCES} PARENT_SCOPE)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories("../libs/" ${CMAKE_CURRENT_SOURCE_DIR}/io)

find_package(BISON)
find_package(FLEX)
set(NETRAX_BISON_FLAGS "-y -d -p rnetwork_")
set(NETRAX_FLEX_FLAGS "-P rnetwork_")
BISON_TARGET(parse_rnetwork_t 
  ${CMAKE_CURRENT_SOURCE_DIR}/io/parse_rnetwork.y ${CMAKE_CURRENT_SOURCE_DIR}/io/parse_rnetwork.c
  COMPILE_FLAGS ${NETRAX_BISON_FLAGS})
FLEX_TARGET(lex_rnetwork_t 
  ${CMAKE_CURRENT_SOURCE_DIR}/io/lex_rnetwork.l ${CMAKE_CURRENT_SOURCE_DIR}/io/lex_rnetwork.c
  COMPILE_FLAGS ${NETRAX_FLEX_FLAGS})
ADD_FLEX_BISON_DEPENDENCY(lex_rnetwork_t parse_rnetwork_t)

#remove annoyting bison and flex warnings
SET_SOURCE_FILES_PROPERTIES( ${FLEX_lex_rnetwork_t_OUTPUTS} PROPERTIES COMPILE_FLAGS -Wno-sign-compare )

set(NETRAX_SOURCES ${PROJECT_SOURCE_DIR}/src/io/lex_rnetwork.c
                   ${PROJECT_SOURCE_DIR}/src/io/parse_rnetwork.c
                   ${NETRAX_SOURCES}
  )

#set(NETRAX_SOURCES ${BISON_parse_rnetwork_t_OUTPUTS}
#                   ${FLEX_lex_rnetwork_t_OUTPUTS}
#                   ${NETRAX_SOURCES}
#  )

# export to the parent dir to use in GTest makefile
set(NETRAX_SOURCES ${NETRAX_SOURCES} PARENT_SCOPE)

add_executable(netrax ${NETRAX_SOURCES})
