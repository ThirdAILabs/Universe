add_subdirectory(layers)
add_subdirectory(networks)
add_subdirectory(utils)
add_subdirectory(tests)

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_network_lib bolt_utils_lib)

enable_testing()
