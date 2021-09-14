enable_testing()

add_executable(
  dwta_test
  DWTATest.cc
)

target_link_libraries(
  dwta_test
  PRIVATE
  dwta_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(dwta_test)