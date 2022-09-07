enable_testing()
include(GoogleTest)

add_library(
  lsh_test_helper_lib
  LSHTestHelper.cc
)

target_link_libraries(
  lsh_test_helper_lib
  PRIVATE
  OpenMP::OpenMP_CXX
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

add_executable(
  densified_minhash_test
  DensifiedMinHashTest.cc
)

set(
  LIBS 
  hash_function_lib
  lsh_test_helper_lib
  gtest_main
  OpenMP::OpenMP_CXX
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

target_link_libraries(
  densified_minhash_test
  PRIVATE
  ${LIBS}
)

set(HASHING_TESTS
  murmurhash_test
  universalhash_test
  srp_test
  fast_srp_test
  densified_minhash_test
)

foreach(HASHING_TEST IN LISTS HASHING_TESTS)
  gtest_discover_tests(${HASHING_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
