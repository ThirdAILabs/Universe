enable_testing()
include(GoogleTest)

add_executable(LicenseTest LicenseTest.cc)

target_link_libraries(LicenseTest PRIVATE cereal::cereal gtest_main ${CRYPTOPP_LIB})

gtest_discover_tests(LicenseTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
