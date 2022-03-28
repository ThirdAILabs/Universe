add_subdirectory(src)

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_network_lib bolt_utils_lib)

add_executable(conv conv.cc)
target_link_libraries(conv PRIVATE bolt_network_lib bolt_utils_lib)

enable_testing()