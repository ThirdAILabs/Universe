enable_testing()

add_executable(metric_test MetricTest.cc)
add_executable(mean_squared_error_test MeanSquaredErrorMetricTest.cc)

set(METRIC_TESTS
  metric_test
  mean_squared_error_test
)

include(GoogleTest)

foreach (METRIC_TEST IN LISTS METRIC_TESTS)
  target_link_libraries(${METRIC_TEST} PRIVATE gtest_main)
endforeach()

foreach (METRIC_TEST IN LISTS METRIC_TESTS)
  gtest_discover_tests(${METRIC_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()


