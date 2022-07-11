add_subdirectory(src)

add_executable(bolt bolt.cc)
target_link_libraries(bolt PRIVATE bolt_network_lib bolt_datasets_lib)

add_executable(TryConv experiments/TryConv.cc)
target_link_libraries(TryConv PRIVATE bolt_network_lib bolt_datasets_lib)

enable_testing()