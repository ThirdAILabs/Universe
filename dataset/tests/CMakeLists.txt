enable_testing()

add_subdirectory(string)
add_subdirectory(bolt_dataset)

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
  block_batch_processor_test
  BlockBatchProcessorTests.cc
)

target_link_libraries(
  block_batch_processor_test
  PRIVATE
  gtest_main
  dataset_core_lib
)

add_executable(
  generic_batch_processor_test
  GenericBatchProcessorTests.cc
)

target_link_libraries(
  generic_batch_processor_test
  PRIVATE
  gtest_main
)

add_executable(
  streaming_generic_dataset_loader_test
  StreamingGenericDatasetLoaderTests.cc
)

target_link_libraries(
  streaming_generic_dataset_loader_test
  PRIVATE
  gtest_main
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
gtest_discover_tests(text_block_test)
gtest_discover_tests(block_batch_processor_test)
gtest_discover_tests(generic_batch_processor_test)
gtest_discover_tests(streaming_generic_dataset_loader_test)
# gtest_discover_tests(global_frequency_test)
