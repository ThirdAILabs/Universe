enable_testing()

add_executable(
  hash_test
  hash_test.cpp
)

target_link_libraries(
  hash_test
  universal_hash_lib
  murmurhash3_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(hash_test)