enable_testing()
include(GoogleTest)

add_executable(
  maxflash_array_test
  MaxFlashArrayTest.cc
)

target_link_libraries(
  maxflash_array_test
  PRIVATE
  hash_function_lib
  gtest_main
)

gtest_discover_tests(maxflash_array_test)
