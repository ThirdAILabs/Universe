enable_testing()

add_executable(fully_connected_network_test FullyConnectedNetworkTest.cc)
target_link_libraries(
  fully_connected_network_test
  PRIVATE
  bolt_network_lib 
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(fully_connected_network_test)