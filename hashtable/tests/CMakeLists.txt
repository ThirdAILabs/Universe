enable_testing()

add_executable(
  vector_hash_table_test
  VectorHashTableTest.cc
)

target_link_libraries(
  vector_hash_table_test
  PRIVATE
  gtest_main
)

target_link_libraries(
  vector_hash_table_test
  PUBLIC
  OpenMP::OpenMP_CXX
)

add_executable(
  sampled_hash_table_test
  SampledHashTableTest.cc
)

target_link_libraries(
  sampled_hash_table_test
  PRIVATE
  gtest_main
)

target_link_libraries(
  sampled_hash_table_test
  PUBLIC
  OpenMP::OpenMP_CXX
)

add_executable(
  tiny_table_test
  TinyTableTest.cc
)

target_link_libraries(
  tiny_table_test
  PRIVATE
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(vector_hash_table_test)
gtest_discover_tests(sampled_hash_table_test)
gtest_discover_tests(tiny_table_test)
