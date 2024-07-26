enable_testing()
include(GoogleTest)

set(THIRDAI_UTILS_TESTS StringManipulationTests SymSpellTests UUIDTests)

foreach(THIRDAI_UTILS_TEST IN LISTS THIRDAI_UTILS_TESTS)
  add_executable(${THIRDAI_UTILS_TEST} ${THIRDAI_UTILS_TEST}.cc)
  target_link_libraries(${THIRDAI_UTILS_TEST} PRIVATE gtest_main thirdai)
  gtest_discover_tests(${THIRDAI_UTILS_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
