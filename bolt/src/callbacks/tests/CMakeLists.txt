enable_testing()

include(GoogleTest)

set(DAG_CALLBACK_TESTS ReduceLROnPlateauTest)

foreach(DAG_CALLBACK_TEST IN LISTS DAG_CALLBACK_TESTS)
  add_executable(${DAG_CALLBACK_TEST} ${DAG_CALLBACK_TEST}.cc)
  target_link_libraries(${DAG_CALLBACK_TEST} PRIVATE thirdai gtest_main
                                                     gmock_main)
  gtest_discover_tests(${DAG_CALLBACK_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
