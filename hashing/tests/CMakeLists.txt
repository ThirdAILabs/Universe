enable_testing()
include(GoogleTest)

add_library(lsh_test_helper_lib LSHTestHelper.cc)

target_link_libraries(lsh_test_helper_lib PUBLIC thirdai gtest_main)

set(HASHING_TESTS MurmurHashTest UniversalHashTest SRPTest FastSRPTest
                  DensifiedMinHashTest MinHashTest DWTATest)

foreach(HASHING_TEST IN LISTS HASHING_TESTS)
  add_executable(${HASHING_TEST} ${HASHING_TEST}.cc)
  target_link_libraries(${HASHING_TEST} PRIVATE lsh_test_helper_lib)
  gtest_discover_tests(${HASHING_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
