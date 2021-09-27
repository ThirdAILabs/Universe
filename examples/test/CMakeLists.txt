enable_testing()

add_executable(
  hash_test
  hash_test.cc
)

target_link_libraries(
  hash_test
  PRIVATE
  universal_hash_lib
  hash_function_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(hash_test)