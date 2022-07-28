enable_testing()

add_executable(
  categorical_block_test
  CategoricalBlockTests.cc
)

target_link_libraries(
  categorical_block_test
  PRIVATE
  gtest_main
)

add_executable(
  text_block_test
  TextBlockTests.cc
)

target_link_libraries(
  text_block_test
  PRIVATE
  gtest_main
  hash_function_lib
)

add_executable(
  categorical_multi_label_test
  CategoricalMultiLabelTest.cc
)

target_link_libraries(
  categorical_multi_label_test
  PRIVATE
  gtest_main
  dataset_core_lib
)

include(GoogleTest)
gtest_discover_tests(categorical_block_test)
gtest_discover_tests(categorical_multi_label_test)
gtest_discover_tests(text_block_test)