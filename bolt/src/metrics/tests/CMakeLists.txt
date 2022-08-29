enable_testing()

add_executable(metric_test MetricTest.cc)
target_link_libraries(
  metric_test
  PRIVATE 
  gtest_main
)

add_executable(mean_squared_error_test MeanSquaredErrorMetricTest.cc)
target_link_libraries(
  mean_squared_error_test
  PRIVATE
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(metric_test DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(mean_squared_error_test DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
