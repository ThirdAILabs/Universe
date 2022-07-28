enable_testing()

add_library(bolt_graph_lib Graph.cc)

target_link_libraries(
  bolt_graph_lib 
  PRIVATE 
  bolt_layer_lib
  license_wrapper_lib
)

add_subdirectory(tests)
