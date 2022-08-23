enable_testing()

add_executable(fully_connected_network_test FullyConnectedNetworkTest.cc)
target_link_libraries(
  fully_connected_network_test
  PRIVATE
  bolt_network_lib 
  gtest_main
)

add_executable(bolt_hash_function_test BoltHashFunctionTest.cc)
target_link_libraries(
  bolt_hash_function_test
  PRIVATE
  bolt_network_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(fully_connected_network_test)
gtest_discover_tests(bolt_hash_function_test)