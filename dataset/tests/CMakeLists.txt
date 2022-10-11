enable_testing()

add_executable(svm_dataset_test SvmDatasetTests.cc)

target_link_libraries(svm_dataset_test PRIVATE gtest_main)

add_executable(click_through_dataset_test ClickThroughDatasetTests.cc)

target_link_libraries(click_through_dataset_test PRIVATE gtest_main)

add_executable(segmented_feature_vectors_test SegmentedFeatureVectorsTests.cc)

target_link_libraries(segmented_feature_vectors_test PRIVATE gtest_main)

add_executable(categorical_block_test CategoricalBlockTests.cc)

target_link_libraries(categorical_block_test PRIVATE gtest_main
                                                     OpenMP::OpenMP_CXX)

add_executable(date_block_test DateBlockTests.cc)

target_link_libraries(date_block_test PRIVATE gtest_main)

target_link_libraries(date_block_test PUBLIC OpenMP::OpenMP_CXX)

add_executable(text_block_test TextBlockTests.cc)

target_link_libraries(text_block_test PRIVATE gtest_main hash_function_lib)

add_executable(quantity_history_tracker_test QuantityHistoryTrackerTest.cc)

target_link_libraries(quantity_history_tracker_test PRIVATE gtest_main
                                                            hash_function_lib)

add_executable(thread_safe_vocabulary_tests ThreadSafeVocabularyTests.cc)

target_link_libraries(thread_safe_vocabulary_tests PRIVATE gtest_main
                                                           OpenMP::OpenMP_CXX)

add_executable(user_item_history_block_tests UserItemHistoryBlockTests.cc)

target_link_libraries(user_item_history_block_tests PRIVATE gtest_main
                                                            OpenMP::OpenMP_CXX)

add_executable(user_count_history_block_tests UserCountHistoryBlockTests.cc)

target_link_libraries(user_count_history_block_tests
                      PRIVATE gtest_main hash_function_lib OpenMP::OpenMP_CXX)

add_executable(block_batch_processor_test BlockBatchProcessorTests.cc)

target_link_libraries(block_batch_processor_test PRIVATE gtest_main
                                                         dataset_core_lib)

add_executable(generic_batch_processor_test GenericBatchProcessorTests.cc)

target_link_libraries(generic_batch_processor_test PRIVATE gtest_main
                                                           OpenMP::OpenMP_CXX)

add_executable(streaming_generic_dataset_loader_tests
               StreamingGenericDatasetLoaderTests.cc)

target_link_libraries(streaming_generic_dataset_loader_tests PRIVATE gtest_main)

target_link_libraries(streaming_generic_dataset_loader_tests
                      PUBLIC OpenMP::OpenMP_CXX)

add_executable(mlm_batch_processor_tests MlmBatchProcessorTests.cc)

target_link_libraries(mlm_batch_processor_tests
                      PRIVATE gtest_main dataset_core_lib hash_function_lib vocab_lib)

target_link_libraries(mlm_batch_processor_tests PUBLIC OpenMP::OpenMP_CXX)

add_executable(save_load_dataset_test SaveLoadInMemoryDatasetTest.cc)
target_link_libraries(save_load_dataset_test PRIVATE gtest_main)

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
  gtest_discover_tests(${DATASET_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
