add_library(bolt_layer_lib ConvLayer.cc EmbeddingLayer.cc FullyConnectedLayer.cc)

target_link_libraries(bolt_layer_lib PUBLIC hash_table_lib hash_function_lib OpenMP::OpenMP_CXX)

add_subdirectory(tests)

enable_testing()
