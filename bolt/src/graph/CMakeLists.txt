enable_testing()

add_library(bolt_graph_lib Graph.cc)

target_link_libraries(bolt_graph_lib PRIVATE bolt_layer_lib license_wrapper_lib)
target_link_libraries(bolt_graph_lib PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(bolt_graph_lib PUBLIC logging_utils_lib)

add_subdirectory(tests)
