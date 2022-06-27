enable_testing()

add_executable(fully_connected_dag_test FullyConnectedDagTest.cc)
target_link_libraries(
  fully_connected_dag_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

add_executable(graph_traversal_test GraphTraversalTest.cc)
target_link_libraries(
  graph_traversal_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

add_executable(graph_reject_inputs_test GraphRejectsInvalidInputsTest.cc)
target_link_libraries(
  graph_reject_inputs_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(fully_connected_dag_test)
gtest_discover_tests(graph_traversal_test)
gtest_discover_tests(graph_reject_inputs_test)
