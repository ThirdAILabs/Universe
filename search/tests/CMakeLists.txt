enable_testing()
include(GoogleTest)

add_executable(MaxFlashArrayTest MaxFlashArrayTest.cc)
add_executable(FlashIndexTest FlashIndexTest.cc)

target_link_libraries(FlashIndexTest PRIVATE thirdai gtest_main)
target_link_libraries(MaxFlashArrayTest PRIVATE thirdai gtest_main)

gtest_discover_tests(FlashIndexTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(MaxFlashArrayTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
