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

add_executable(bolt_layer_test_utils_test BoltLayerTestUtilsTest.cc)
target_link_libraries(
  bolt_layer_test_utils_test
  PRIVATE
  gtest_main
)

include(GoogleTest)

set(BOLT_LAYER_TESTS
  fully_connected_layer_test
  embedding_layer_test
  bolt_layer_test_utils_test
)

foreach(BOLT_LAYER_TEST IN LISTS BOLT_LAYER_TESTS)
  gtest_discover_tests(${BOLT_LAYER_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
