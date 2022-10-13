
enable_testing()

include(GoogleTest)

add_executable(
  flash_index_test
  FlashIndexTest.cc
)

target_link_libraries(
  flash_index_test
  PRIVATE
  hash_function_lib
  flash_indexer_lib
  gtest_main
)

gtest_discover_tests(flash_index_test)
