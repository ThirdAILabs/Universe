enable_testing()

include(GoogleTest)

set(BOLT_LAYER_TESTS FullyConnectedLayerTest EmbeddingLayerTest
                     BoltLayerTestUtilsTest)

foreach(BOLT_LAYER_TEST IN LISTS BOLT_LAYER_TESTS)
  add_executable(${BOLT_LAYER_TEST} ${BOLT_LAYER_TEST}.cc)
  target_link_libraries(${BOLT_LAYER_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${BOLT_LAYER_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
