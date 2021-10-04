enable_testing()
include(GoogleTest)

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
  srp_test
  SRPTest.cc
)

add_executable(
  fast_srp_test
  FastSRPTest.cc
)

add_executable(
  densified_minhash_test
  DensifiedMinHashTest.cc
)

target_link_libraries(
  srp_test
  PRIVATE
  hash_function_lib
  lsh_test_helper_lib
  gtest_main
)

target_link_libraries(
  fast_srp_test
  PRIVATE
  hash_function_lib
  lsh_test_helper_lib
  gtest_main
)

target_link_libraries(
  densified_minhash_test
  PRIVATE
  hash_function_lib
  lsh_test_helper_lib
  gtest_main
)

gtest_discover_tests(srp_test)
gtest_discover_tests(fast_srp_test)
gtest_discover_tests(densified_minhash_test)