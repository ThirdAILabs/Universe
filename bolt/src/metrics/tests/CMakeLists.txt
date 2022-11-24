enable_testing()

include(GoogleTest)

set(METRIC_TESTS MetricTest MeanSquaredErrorMetricTest
                 CategoricalCrossEntropyTest)

foreach(METRIC_TEST IN LISTS METRIC_TESTS)
  add_executable(${METRIC_TEST} ${METRIC_TEST}.cc)
  target_link_libraries(${METRIC_TEST} PRIVATE gtest_main thirdai)
  gtest_discover_tests(${METRIC_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
