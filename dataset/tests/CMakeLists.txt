enable_testing()

add_subdirectory(blocks)

add_executable(
  svm_dataset_test
  SvmDatasetTests.cc
)

target_link_libraries(
  svm_dataset_test
  PRIVATE
  gtest_main
)

add_executable(
  click_through_dataset_test
  ClickThroughDatasetTests.cc
)

target_link_libraries(
  click_through_dataset_test
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
  streaming_generic_dataset_loader_tests
  StreamingGenericDatasetLoaderTests.cc
)

target_link_libraries(
  streaming_generic_dataset_loader_tests
  PRIVATE
  gtest_main
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

add_executable(
  text_processing_tests
  TextProcessingTests.cc
)

target_link_libraries(
  text_processing_tests
  PRIVATE
  gtest_main
  dataset_core_lib
  hash_function_lib
)

include(GoogleTest)
gtest_discover_tests(svm_dataset_test)
gtest_discover_tests(click_through_dataset_test)
gtest_discover_tests(segmented_feature_vectors_test)
gtest_discover_tests(block_batch_processor_test)
gtest_discover_tests(generic_batch_processor_test)
gtest_discover_tests(streaming_generic_dataset_loader_tests)
gtest_discover_tests(categorical_multi_label_test)
gtest_discover_tests(text_processing_tests)
