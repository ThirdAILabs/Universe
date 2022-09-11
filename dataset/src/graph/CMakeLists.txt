enable_testing()

add_executable(
  graph_batch_processor_test
  GraphBatchProcessorTest.cc
)

target_link_libraries(
  graph_batch_processor_test
  PRIVATE
  gtest_main
  OpenMP::OpenMP_CXX
)

include(GoogleTest)

set(GRAPH_DATASET_TESTS
  graph_batch_processor_test
)

foreach(GRAPH_DATASET_TEST IN LISTS GRAPH_DATASET_TESTS)
  gtest_discover_tests(${GRAPH_DATASET_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
