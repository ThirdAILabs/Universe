enable_testing()
include(GoogleTest)

set(SEARCH_TESTS MaxFlashArrayTest FlashIndexTest InvertedIndexTests)

foreach(SEARCH_TEST IN LISTS SEARCH_TESTS)
  add_executable(${SEARCH_TEST} ${SEARCH_TEST}.cc)
  target_link_libraries(${SEARCH_TEST} PRIVATE thirdai gtest_main)

  gtest_discover_tests(${SEARCH_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})

endforeach(SEARCH_TEST IN LISTS SEARCH_TESTS)
