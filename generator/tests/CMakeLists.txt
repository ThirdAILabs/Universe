enable_testing()
include(GoogleTest)

add_executable(FlashGeneratorTest FlashGeneratorTest.cc)

target_link_libraries(FlashGeneratorTest PRIVATE thirdai gmock_main gtest_main)

set(FLASH_GENERATOR_TESTS FlashGeneratorTest)

foreach(FLASH_GENERATOR_TEST IN LISTS FLASH_GENERATOR_TESTS)
  gtest_discover_tests(${FLASH_GENERATOR_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
