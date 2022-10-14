
enable_testing()


add_executable(
  flash_index_test
  FlashIndexTest.cc
)

target_link_libraries(
  flash_index_test
  PRIVATE
  hash_function_lib
  flash_indexer_lib
  gtest_main
)

include(GoogleTest)

set(FLASH_INDEX_TESTS flash_index_test)

foreach(FLASH_INDEX_TEST IN LISTS FLASH_INDEX_TESTS)
  gtest_discover_tests(${FLASH_INDEX_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()


