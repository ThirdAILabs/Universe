enable_testing()

add_executable(
  vector_hash_table_test
  VectorHashTableTest.cc
)

target_link_libraries(
  vector_hash_table_test
  PRIVATE
  hash_table_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(vector_hash_table_test)