enable_testing()

add_executable(
  categorical_block_test
  CategoricalBlockTests.cc
)

target_link_libraries(
  categorical_block_test
  PRIVATE
  gtest_main
)

add_executable(
  date_block_test
  DateBlockTests.cc
)

target_link_libraries(
  date_block_test
  PRIVATE
  gtest_main
)

add_executable(
  text_block_test
  TextBlockTests.cc
)

target_link_libraries(
  text_block_test
  PRIVATE
  gtest_main
  hash_function_lib
)

add_executable(
  trend_block_test
  TrendBlockTests.cc
)

target_link_libraries(
  trend_block_test
  PRIVATE
  gtest_main
  hash_function_lib
)

add_executable(
  categorical_tracking_block_test
  CategoricalTrackingBlockTests.cc
)

target_link_libraries(
  categorical_tracking_block_test
  PRIVATE
  gtest_main
)


include(GoogleTest)
gtest_discover_tests(categorical_block_test)
gtest_discover_tests(date_block_test)
gtest_discover_tests(text_block_test)
gtest_discover_tests(trend_block_test)
gtest_discover_tests(categorical_tracking_block_test)
