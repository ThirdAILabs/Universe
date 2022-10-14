enable_testing()

add_executable(SequentialClassifierTest SequentialClassifierTest.cc)
target_link_libraries(SequentialClassifierTest PRIVATE thirdai gtest_main)

include(GoogleTest)

gtest_discover_tests(SequentialClassifierTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
