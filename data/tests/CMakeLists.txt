enable_testing()

include(GoogleTest)

set(DATA_TESTS
    TensorConversionTests
    DataLoaderTest
    PermutationTest
    EncodePositionTest
    CountTokensTest
    RecurrenceTest
    GraphTests
    ExplanationTests
    RegressionBinningTest
    DeduplicateTokensTest
    ColumnMapIteratorTests
    TabularTests
    StateSerializationTest
    TextCompatTest
    MachMemorySerializationTest
    NerRuleTests)

foreach(DATA_TEST IN LISTS DATA_TESTS)
  add_executable(${DATA_TEST} ${DATA_TEST}.cc)
  target_link_libraries(${DATA_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${DATA_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
