enable_testing()

add_executable(svm_dataset_test SvmDatasetTests.cc)

add_executable(click_through_dataset_test ClickThroughDatasetTests.cc)

add_executable(segmented_feature_vectors_test SegmentedFeatureVectorsTests.cc)

add_executable(categorical_block_test CategoricalBlockTests.cc)

add_executable(date_block_test DateBlockTests.cc)

add_executable(text_block_test TextBlockTests.cc)

add_executable(quantity_history_tracker_test QuantityHistoryTrackerTest.cc)

add_executable(thread_safe_vocabulary_tests ThreadSafeVocabularyTests.cc)

add_executable(user_item_history_block_tests UserItemHistoryBlockTests.cc)

add_executable(user_count_history_block_tests UserCountHistoryBlockTests.cc)

add_executable(block_batch_processor_test BlockBatchProcessorTests.cc)

add_executable(generic_batch_processor_test GenericBatchProcessorTests.cc)

add_executable(streaming_generic_dataset_loader_tests
               StreamingGenericDatasetLoaderTests.cc)

add_executable(mlm_batch_processor_tests MlmBatchProcessorTests.cc)

add_executable(save_load_dataset_test SaveLoadInMemoryDatasetTest.cc)

include(GoogleTest)

set(DATASET_TESTS
    svm_dataset_test
    click_through_dataset_test
    segmented_feature_vectors_test
    categorical_block_test
    date_block_test
    text_block_test
    quantity_history_tracker_test
    thread_safe_vocabulary_tests
    user_item_history_block_tests
    user_count_history_block_tests
    block_batch_processor_test
    generic_batch_processor_test
    mlm_batch_processor_tests
    save_load_dataset_test
    streaming_generic_dataset_loader_tests)

foreach(DATASET_TEST IN LISTS DATASET_TESTS)
  target_link_libraries(${DATASET_TEST} PRIVATE gtest_main thirdai)

  gtest_discover_tests(${DATASET_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
