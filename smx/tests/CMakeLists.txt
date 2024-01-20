enable_testing()

include(GoogleTest)

set(SMX_TESTS
    AutogradTests
    ModuleTests)

foreach(SMX_TEST IN LISTS SMX_TESTS)
  add_executable(${SMX_TEST} ${SMX_TEST}.cc)
  target_link_libraries(${SMX_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${SMX_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
