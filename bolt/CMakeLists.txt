file(GLOB SOURCES src/*.cc)

add_library(bolt_lib STATIC ${SOURCES})
target_link_libraries(bolt_lib PRIVATE hash_function_lib sampled_hash_table_lib)

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_lib)

enable_testing()

add_subdirectory(tests)