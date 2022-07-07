enable_testing()

add_executable(sequential_classifier_tests SequentialClassifierTests.cc)
target_link_libraries(
  sequential_classifier_tests 
  PRIVATE 
  sequential_classifier_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(sequential_classifier_tests)
