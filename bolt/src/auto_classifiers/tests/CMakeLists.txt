enable_testing()

add_executable(bolt_tabular_classifier_test TabularClassifierTest.cc)
target_link_libraries(
  bolt_tabular_classifier_test
  PRIVATE
  auto_classifier_lib
  bolt_network_lib
  gtest_main
)

add_executable(bolt_text_classifier_test TextClassifierTest.cc)
target_link_libraries(
  bolt_text_classifier_test
  PRIVATE
  auto_classifier_lib
  bolt_network_lib
  gtest_main
)

add_executable(bolt_multi_label_text_classifier_test MultiLabelTextClassifierTest.cc)
target_link_libraries(
  bolt_multi_label_text_classifier_test
  PRIVATE
  auto_classifier_lib
  bolt_network_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(bolt_tabular_classifier_test)
gtest_discover_tests(bolt_text_classifier_test)
gtest_discover_tests(bolt_multi_label_text_classifier_test)