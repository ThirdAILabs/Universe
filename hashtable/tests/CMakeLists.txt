enable_testing()

add_executable(vector_hash_table_test VectorHashTableTest.cc)

target_link_libraries(vector_hash_table_test PRIVATE hash_table_lib gtest_main)

add_executable(sampled_hash_table_test SampledHashTableTest.cc)

target_link_libraries(sampled_hash_table_test PRIVATE hash_table_lib gtest_main)

add_executable(tiny_table_test TinyTableTest.cc)

target_link_libraries(tiny_table_test PRIVATE hash_table_lib gtest_main)

include(GoogleTest)

set(HASH_TABLE_TESTS vector_hash_table_test sampled_hash_table_test
                     tiny_table_test)

foreach(HASH_TABLE_TEST IN LISTS HASH_TABLE_TESTS)
  gtest_discover_tests(${HASH_TABLE_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
