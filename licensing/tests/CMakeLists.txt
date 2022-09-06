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
  cryptopp
)

gtest_discover_tests(license_test DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
