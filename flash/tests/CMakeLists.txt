enable_testing()
include(GoogleTest)

add_executable(
  flash_test
  FlashTest.cc
)

target_link_libraries(
  flash_test
  PRIVATE
  hash_function_lib
  flash_lib
  gtest_main
  dataset_lib
)

gtest_discover_tests(flash_test)
