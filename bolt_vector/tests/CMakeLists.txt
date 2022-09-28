enable_testing()

add_executable(bolt_vector_test BoltVectorTest.cc)
target_link_libraries(bolt_vector_test PRIVATE gtest_main)

add_executable(bolt_vector_save_load_test BoltVectorLoadSaveTest.cc)
target_link_libraries(bolt_vector_save_load_test PRIVATE gtest_main)

add_executable(bolt_vector_norm_test BoltVectorNormTest.cc)
target_link_libraries(bolt_vector_norm_test PRIVATE gmock_main gtest_main)

include(GoogleTest)

set(BOLT_VECTOR_TESTS bolt_vector_test bolt_vector_save_load_test
                      bolt_vector_norm_test)

foreach(BOLT_VECTOR_TEST IN LISTS BOLT_VECTOR_TESTS)
  gtest_discover_tests(${BOLT_VECTOR_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
