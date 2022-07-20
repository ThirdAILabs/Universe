enable_testing()

add_executable(
  text_processing_tests
  TextProcessingTests.cc
)

target_link_libraries(
  text_processing_tests
  PRIVATE
  hash_function_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(text_processing_tests)
