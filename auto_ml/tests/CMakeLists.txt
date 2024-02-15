enable_testing()
include(GoogleTest)

set(AUTOML_TESTS ConfigParameterTests TabularTransformationTests
                 BalancingSamplesSerializationTest)

foreach(AUTOML_TEST IN LISTS AUTOML_TESTS)
  add_executable(${AUTOML_TEST} ${AUTOML_TEST}.cc)
  target_link_libraries(${AUTOML_TEST} PRIVATE thirdai gtest_main)

  gtest_discover_tests(${AUTOML_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})

endforeach(AUTOML_TEST IN LISTS AUTOML_TESTS)
