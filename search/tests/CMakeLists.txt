enable_testing()
include(GoogleTest)

add_executable(
  flash_test
  FlashTest.cc
)

add_executable(
  maxflash_array_test
  MaxFlashArrayTest.cc
)

target_link_libraries(
  flash_test
  PRIVATE
  hash_function_lib
  flash_lib
  gtest_main
)

target_link_libraries(
  maxflash_array_test
  PRIVATE
  hash_function_lib
  maxflash_lib
  gtest_main
)

gtest_discover_tests(flash_test)
gtest_discover_tests(maxflash_array_test)
