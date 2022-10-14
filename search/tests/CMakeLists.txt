enable_testing()
include(GoogleTest)

add_executable(MaxFlashArrayTest MaxFlashArrayTest.cc)

target_link_libraries(MaxFlashArrayTest PRIVATE thirdai gtest_main)

gtest_discover_tests(MaxFlashArrayTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
