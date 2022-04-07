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

add_executable(bolt_vector_test BoltVectorTest.cc)
target_link_libraries(
  bolt_vector_test
  PRIVATE
  gtest_main
)

add_executable(bolt_layer_test_utils_test BoltLayerTestUtilsTest.cc)
target_link_libraries(
  bolt_layer_test_utils_test
  PRIVATE
  gtest_main
)


include(GoogleTest)
gtest_discover_tests(fully_connected_layer_test)
gtest_discover_tests(embedding_layer_test)
gtest_discover_tests(conv_layer_test)
gtest_discover_tests(bolt_vector_test)
gtest_discover_tests(bolt_layer_test_utils_test)