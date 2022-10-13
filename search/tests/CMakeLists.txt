enable_testing()
include(GoogleTest)

add_executable(maxflash_array_test MaxFlashArrayTest.cc)

target_link_libraries(maxflash_array_test PRIVATE thirdai gtest_main)

gtest_discover_tests(maxflash_array_test DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
