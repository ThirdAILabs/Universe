enable_testing()

add_subdirectory(string)

add_executable(
  svm_dataset_test
  SvmDatasetTests.cc
)

target_link_libraries(
  svm_dataset_test
  PRIVATE
  gtest_main
  bolt_datasets_lib
)

add_executable(
  csv_dataset_test
  CsvDatasetTests.cc
)

target_link_libraries(
  csv_dataset_test
  PRIVATE
  gtest_main
  bolt_datasets_lib
)

add_executable(
  click_through_dataset_test
  ClickThroughDatasetTests.cc
)

target_link_libraries(
  click_through_dataset_test
  PRIVATE
  gtest_main
  bolt_datasets_lib
)

add_executable(
  vector_test
  VectorTest.cc
)

target_link_libraries(
  vector_test
  PRIVATE
  gtest_main
)

add_executable(
  segmented_feature_vectors_test
  SegmentedFeatureVectorsTests.cc
)

target_link_libraries(
  segmented_feature_vectors_test
  PRIVATE
  gtest_main
)

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
  batch_processor_test
  BatchProcessorTests.cc
)

target_link_libraries(
  batch_processor_test
  PRIVATE
  gtest_main
  dataset_core_lib
)

# add_executable(
#   global_frequency_test
#   GlobalFreqTest.cc
# )

# target_link_libraries(
#   global_frequency_test
#   PRIVATE
#   global_frequency_lib
#   gtest_main
# )

include(GoogleTest)
gtest_discover_tests(svm_dataset_test)
gtest_discover_tests(click_through_dataset_test)
gtest_discover_tests(vector_test)
gtest_discover_tests(csv_dataset_test)
gtest_discover_tests(segmented_feature_vectors_test)
gtest_discover_tests(categorical_block_test)
gtest_discover_tests(batch_processor_test)
# gtest_discover_tests(global_frequency_test)
