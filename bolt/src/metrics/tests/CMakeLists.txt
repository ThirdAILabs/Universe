enable_testing()

add_executable(metric_test MetricTest.cc)
target_link_libraries(
  metric_test
  PRIVATE 
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(metric_test)