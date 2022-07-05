enable_testing()

add_library(bolt_graph_lib Graph.cc)

target_link_libraries(
  bolt_graph_lib 
  PRIVATE 
  bolt_network_lib
)

add_subdirectory(tests)
