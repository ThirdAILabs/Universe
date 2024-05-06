enable_testing()

include(GoogleTest)

set(MACH_TESTS MachTrainerTests)

foreach(MACH_TEST IN LISTS MACH_TESTS)
  add_executable(${MACH_TEST} ${MACH_TEST}.cc)
  target_link_libraries(${MACH_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${MACH_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
