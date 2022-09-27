
enable_testing()

add_executable(sequential_classifier_data_type_test SequentialClassifierDataTypeTest.cc)


target_link_libraries(sequential_classifier_data_type_test PRIVATE gtest_main OpenMP::OpenMP_CXX)

set(SEQUENTIAL_CLASSIFIER_DATA_TYPE_TESTS sequential_classifier_data_type_test)

include(GoogleTest)
foreach(TEST IN LISTS SEQUENTIAL_CLASSIFIER_DATA_TYPE_TESTS)
    gtest_discover_tests(${TEST} DISCOVERY_TIMEOUT
                         ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
