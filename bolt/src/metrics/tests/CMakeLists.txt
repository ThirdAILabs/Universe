enable_testing()

add_executable(metric_test MetricTest.cc)
target_link_libraries(
  metric_test
  PRIVATE 
  gtest_main
)

add_executable(mse_test MeanSquaredErrorMetricTest.cc)
target_link_libraries(
  mse_test
  PRIVATE 
  gtest_main
)

SET(METRIC_TESTS
  metric_test
  mse_test
)

include(GoogleTest)
foreach(METRIC_TEST IN LISTS METRIC_TESTS)
  gtest_discover_tests(${METRIC_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
