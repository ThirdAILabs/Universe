file(GLOB SOURCES src/*.cc)

add_library(bolt_lib STATIC ${SOURCES})
target_link_libraries(bolt_lib PRIVATE dwta_lib)

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_lib)