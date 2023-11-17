enable_testing()

set(DATASET_TESTS
    CsvParserTests
    CsvDataSourceTests
    SvmDatasetTests
    ClickThroughDatasetTests
    SegmentedFeatureVectorsTests
    CategoricalBlockTests
    DateBlockTests
    TextBlockTests
    MachBlockTests
    MachDecodeTests
    MachIndexSerializationTests
    QuantityHistoryTrackerTest
    ThreadSafeVocabularyTests
    UserItemHistoryBlockTests
    UserCountHistoryBlockTests
    TabularFeaturizerTests
    SaveLoadInMemoryDatasetTest
    DatasetLoaderTests
    TokenEncodingTests
    LLMFeaturizerTests
    RecurrenceAugmentationTests)

include(GoogleTest)

foreach(DATASET_TEST IN LISTS DATASET_TESTS)
  add_executable(${DATASET_TEST} ${DATASET_TEST}.cc)
  target_link_libraries(${DATASET_TEST} PRIVATE gtest_main thirdai)

  gtest_discover_tests(${DATASET_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
