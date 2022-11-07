enable_testing()

include(GoogleTest)

set(COMPRESSION_TESTS DragonVectorTest CountSketchTest)

foreach(COMPRESSION_TEST IN LISTS COMPRESSION_TESTS)
  add_executable(${COMPRESSION_TEST} ${COMPRESSION_TEST}.cc)
  target_link_libraries(${COMPRESSION_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${COMPRESSION_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
