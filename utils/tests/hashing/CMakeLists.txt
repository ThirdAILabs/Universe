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
  dwta_test
  DWTATest.cc
)

target_link_libraries(
  dwta_test
  PRIVATE
  dwta_lib
  lsh_test_helper_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(dwta_test)