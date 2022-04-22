enable_testing()
include(GoogleTest)

add_executable(
  number_test
  NumberTest.cc
)

add_executable(
  feature_hashing_test
  FeatureHashingTest.cc
)

add_executable(
  text_test
  TextTest.cc
)

add_executable(
  dynamic_counts_test
  DynamicCountsTest.cc
)

target_link_libraries(
  number_test
  PRIVATE
  gtest_main
)

target_link_libraries(
  feature_hashing_test
  PRIVATE
  gtest_main
)

target_link_libraries(
  text_test
  PRIVATE
  gtest_main
)

target_link_libraries(
  dynamic_counts_test
  PRIVATE
  gtest_main
)

gtest_discover_tests(number_test)
gtest_discover_tests(feature_hashing_test)
gtest_discover_tests(text_test)
gtest_discover_tests(dynamic_counts_test)