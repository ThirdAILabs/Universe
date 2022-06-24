enable_testing()

add_executable(fully_connected_dag_test FullyConnectedDagTest.cc)
target_link_libraries(
  fully_connected_dag_test
  PRIVATE
  bolt_graph_lib 
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(fully_connected_dag_test)
