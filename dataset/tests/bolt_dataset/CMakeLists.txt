enable_testing()

add_executable(
  text_classifier_batch_processor_tests
  TextClassifierBatchProcessorTests.cc
)

target_link_libraries(
  text_classifier_batch_processor_tests
  PRIVATE
  bolt_datasets_lib
  hash_function_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(text_classifier_batch_processor_tests)
