enable_testing()

add_library(bolt_graph_lib Graph.cc DistributedBoltGraph.cc)

target_link_libraries(
  bolt_graph_lib 
  PRIVATE 
  bolt_layer_lib
  license_wrapper_lib
)
target_link_libraries(
    bolt_graph_lib
    PUBLIC
    OpenMP::OpenMP_CXX)

add_subdirectory(tests)
