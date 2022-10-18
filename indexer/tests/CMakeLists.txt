enable_testing()
include(GoogleTest)

add_executable(FlashIndexTest FlashIndexTest.cc)

target_link_libraries(FlashIndexTest PRIVATE thirdai gmock_main gtest_main)

set(FLASH_INDEX_TESTS FlashIndexTest)

foreach(FLASH_INDEX_TEST IN LISTS FLASH_INDEX_TESTS)
  gtest_discover_tests(${FLASH_INDEX_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()

