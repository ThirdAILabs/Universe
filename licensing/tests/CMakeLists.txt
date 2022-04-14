enable_testing()
include(GoogleTest)

add_executable(
  license_test
  LicenseTest.cc
)

target_link_libraries(
  license_test
  PRIVATE
  gtest_main
  license_lib
  libcryptopp
)

gtest_discover_tests(license_test)
