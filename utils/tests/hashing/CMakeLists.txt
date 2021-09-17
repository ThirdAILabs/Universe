enable_testing()

add_library(
  lsh_test_helper_lib
  LSHTestHelper.cc
)

target_link_libraries(
  lsh_test_helper_lib
  PRIVATE
  gtest_main
)

add_executable(
  cosine_test
  SRPTest.cc
)

target_link_libraries(
  cosine_test
  PRIVATE
  hash_function_lib
  lsh_test_helper_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(cosine_test)