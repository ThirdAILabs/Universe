enable_testing()

add_executable(
  hash_function_test
  HashFunctionTest.cc
)

target_link_libraries(
  hash_function_test
  PRIVATE
  hash_function_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(hash_function_test)