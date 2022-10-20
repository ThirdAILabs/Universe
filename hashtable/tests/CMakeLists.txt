enable_testing()

include(GoogleTest)

set(HASH_TABLE_TESTS VectorHashTableTest SampledHashTableTest TinyTableTest)

foreach(HASH_TABLE_TEST IN LISTS HASH_TABLE_TESTS)
  add_executable(${HASH_TABLE_TEST} ${HASH_TABLE_TEST}.cc)
  target_link_libraries(${HASH_TABLE_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${HASH_TABLE_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
