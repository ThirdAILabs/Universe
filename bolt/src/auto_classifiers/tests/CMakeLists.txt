enable_testing()

add_executable(bolt_tabular_classifier_test TabularClassifierTest.cc)
target_link_libraries(
  bolt_tabular_classifier_test
  PRIVATE
  auto_classifier_lib
  gtest_main
)

add_executable(bolt_text_classifier_test TextClassifierTest.cc)
target_link_libraries(
  bolt_text_classifier_test
  PRIVATE
  auto_classifier_lib
  gtest_main
)

add_executable(bolt_multi_label_text_classifier_test MultiLabelTextClassifierTest.cc)
target_link_libraries(
  bolt_multi_label_text_classifier_test
  PRIVATE
  auto_classifier_lib
  gtest_main
)

add_executable(bolt_sequential_classifier_test SequentialClassifierTest.cc)
target_link_libraries(
  bolt_sequential_classifier_test
  PRIVATE
  auto_classifier_lib
  gtest_main
)

include(GoogleTest)

set(AUTO_CLASSIFIER_TESTS
  bolt_tabular_classifier_test
  bolt_text_classifier_test
  bolt_multi_label_text_classifier_test
  bolt_sequential_classifier_test
)

foreach(AUTO_CLASSIFIER_TEST IN LISTS AUTO_CLASSIFIER_TESTS)
  gtest_discover_tests(${AUTO_CLASSIFIER_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
