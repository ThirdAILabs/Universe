enable_testing()

add_executable(bolt_sequential_classifier_test SequentialClassifierTest.cc)
target_link_libraries(bolt_sequential_classifier_test PRIVATE thirdai
                                                              gtest_main)

include(GoogleTest)

gtest_discover_tests(bolt_sequential_classifier_test DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
