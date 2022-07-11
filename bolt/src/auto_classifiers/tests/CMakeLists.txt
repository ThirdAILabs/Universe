enable_testing()

add_executable(bolt_tabular_classifier_test TabularClassifierTest.cc)
target_link_libraries(
  bolt_tabular_classifier_test
  PRIVATE
  bolt_network_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(bolt_tabular_classifier_test)