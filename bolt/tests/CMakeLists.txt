enable_testing()

add_executable(fully_connected_layer_test FullyConnectedLayerTest.cc)
target_link_libraries(
  fully_connected_layer_test 
  PRIVATE 
  bolt_layer_lib
  gtest_main
)

add_executable(embedding_layer_test EmbeddingLayerTest.cc)
target_link_libraries(
  embedding_layer_test 
  PRIVATE 
  bolt_layer_lib
  gtest_main
)


include(GoogleTest)
gtest_discover_tests(fully_connected_layer_test)
gtest_discover_tests(embedding_layer_test)