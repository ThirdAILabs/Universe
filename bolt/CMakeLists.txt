file(GLOB SOURCES src/*.cc)

add_library(bolt_lib STATIC ${SOURCES})

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_lib)