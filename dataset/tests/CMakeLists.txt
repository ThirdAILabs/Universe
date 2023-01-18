enable_testing()

set(DATASET_TESTS
    SvmDatasetTests
    ClickThroughDatasetTests
    SegmentedFeatureVectorsTests
    CategoricalBlockTests
    DateBlockTests
    TextBlockTests
    QuantityHistoryTrackerTest
    ThreadSafeVocabularyTests
    UserItemHistoryBlockTests
    UserCountHistoryBlockTests
    GenericBatchProcessorTests
    MlmBatchProcessorTests
    SaveLoadInMemoryDatasetTest
    DatasetLoaderTests
    TokenEncodingTests)

include(GoogleTest)

foreach(DATASET_TEST IN LISTS DATASET_TESTS)
  add_executable(${DATASET_TEST} ${DATASET_TEST}.cc)
  target_link_libraries(${DATASET_TEST} PRIVATE gtest_main thirdai)

  gtest_discover_tests(${DATASET_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
