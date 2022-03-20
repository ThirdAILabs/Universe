enable_testing()
include(GoogleTest)

add_executable(
  flash_test
  FlashTest.cc
)

add_executable(
  max_flash_array_test
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
  max_flash_array_test
  PRIVATE
  hash_function_lib
  flash_lib
  gtest_main
)

gtest_discover_tests(flash_test)
gtest_discover_tests(max_flash_array_test)
