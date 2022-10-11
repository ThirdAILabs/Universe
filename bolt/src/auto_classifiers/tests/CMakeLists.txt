enable_testing()

add_executable(bolt_sequential_classifier_test SequentialClassifierTest.cc)
target_link_libraries(bolt_sequential_classifier_test PRIVATE thirdai
                                                              gtest_main)

include(GoogleTest)

set(AUTO_CLASSIFIER_TESTS bolt_sequential_classifier_test)

foreach(AUTO_CLASSIFIER_TEST IN LISTS AUTO_CLASSIFIER_TESTS)
  gtest_discover_tests(${AUTO_CLASSIFIER_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
