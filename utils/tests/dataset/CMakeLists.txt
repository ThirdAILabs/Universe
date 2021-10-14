enable_testing()

add_executable(
  svm_dataset_test
  SvmDatasetTests.cc
)

target_link_libraries(
  svm_dataset_test
  PRIVATE
  dataset_lib
  gtest_main
)

add_executable(
  click_through_dataset_test
  ClickThroughDatasetTests.cc
)

target_link_libraries(
  click_through_dataset_test
  PRIVATE
  dataset_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(svm_dataset_test)
gtest_discover_tests(click_through_dataset_test)
