enable_testing()

add_executable(LossFunctionGradientTest LossFunctionGradientTest.cc)
target_link_libraries(LossFunctionGradientTest PRIVATE gtest_main thirdai)

include(GoogleTest)
gtest_discover_tests(LossFunctionGradientTest DISCOVERY_TIMEOUT
                     ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
