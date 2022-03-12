enable_testing()

add_executable(loss_function_test LossFunctionTest.cc)
target_link_libraries(
  loss_function_test
  PRIVATE 
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(loss_function_test)