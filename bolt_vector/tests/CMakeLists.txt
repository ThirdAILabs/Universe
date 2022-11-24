enable_testing()

include(GoogleTest)

set(BOLT_VECTOR_TESTS BoltVectorTest BoltVectorLoadSaveTest BoltVectorNormTest)

foreach(BOLT_VECTOR_TEST IN LISTS BOLT_VECTOR_TESTS)
  add_executable(${BOLT_VECTOR_TEST} ${BOLT_VECTOR_TEST}.cc)
  target_link_libraries(${BOLT_VECTOR_TEST} PRIVATE gtest_main thirdai)
  gtest_discover_tests(${BOLT_VECTOR_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
