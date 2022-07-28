enable_testing()

add_executable(metric_test MetricTest.cc)
target_link_libraries(
  metric_test
  PRIVATE 
  gtest_main
)

target_link_libraries(
    metric_test
    PUBLIC
    OpenMP::OpenMP_CXX)

include(GoogleTest)
gtest_discover_tests(metric_test)
