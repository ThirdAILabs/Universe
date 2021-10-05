enable_testing()
include(GoogleTest)

add_library(
  lsh_test_helper_lib
  LSHTestHelper.cc
)

add_library(
  gen_rand_keys_test_suite_lib
  AvalancheTimedTestSuite.h
)

target_link_libraries(
  lsh_test_helper_lib
  PRIVATE
  gtest_main
)

target_link_libraries(
  gen_rand_keys_test_suite_lib
  PRIVATE
  gtest_main
)

add_executable(
  murmurhash_test
  MurmurHashTest.cc
)

add_executable(
  universalhash_test
  UniversalHashTest.cc
)

add_executable(
  srp_test
  SRPTest.cc
)

add_executable(
  fast_srp_test
  FastSRPTest.cc
)

set(
  LIBS 
  hash_function_lib
  lsh_test_helper_lib
  gen_rand_keys_test_suite_lib
  gtest_main
)

target_link_libraries(
  srp_test
  PRIVATE
  ${LIBS}
)

target_link_libraries(
  fast_srp_test
  PRIVATE
  ${LIBS}
)

target_link_libraries(
  murmurhash_test
  PRIVATE
  ${LIBS}
)

target_link_libraries(
  universalhash_test
  PRIVATE
  ${LIBS}
)

gtest_discover_tests(murmurhash_test)
gtest_discover_tests(universalhash_test)
gtest_discover_tests(srp_test)
gtest_discover_tests(fast_srp_test)