enable_testing()
include(GoogleTest)

add_executable(QueryCandidateGeneratorConfigTest QueryCandidateGeneratorTest.cc)

target_link_libraries(QueryCandidateGeneratorConfigTest
                      PRIVATE thirdai gmock_main gtest_main)

gtest_discover_tests(QueryCandidateGeneratorConfigTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
