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

include(GoogleTest)
gtest_discover_tests(svm_dataset_test)
