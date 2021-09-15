enable_testing()

add_executable(
  vector_hash_table_test
  VectorHashTableTest.cc
)

find_package(OpenMP REQUIRED)

target_link_libraries(
  vector_hash_table_test
  PRIVATE
  hash_table_lib
  gtest_main
  OpenMP::OpenMP_CXX
)

add_executable(
  sampled_hash_table_test
  SampledHashTableTest.cc
)

target_link_libraries(
  sampled_hash_table_test
  PRIVATE
  sampled_hash_table_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(vector_hash_table_test)
gtest_discover_tests(sampled_hash_table_test)