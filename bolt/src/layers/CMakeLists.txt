add_library(bolt_layer_lib EmbeddingLayer.cc FullyConnectedLayer.cc)

target_link_libraries(bolt_layer_lib hash_table_lib hash_function_lib)

add_subdirectory(tests)

enable_testing()