enable_testing()

add_executable(loss_function_gradient_test LossFunctionGradientTest.cc)
target_link_libraries(
  loss_function_gradient_test
  PRIVATE 
  gtest_main
)

add_executable(loss_function_metric_test LossFunctionMetricTest.cc)
target_link_libraries(
  loss_function_metric_test
  PRIVATE 
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(loss_function_gradient_test)
gtest_discover_tests(loss_function_metric_test)