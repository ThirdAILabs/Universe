enable_testing()

add_executable(layer_norm_node_test LayerNormNodeTest.cc)
target_link_libraries(
  layer_norm_node_test
  PRIVATE
  bolt_graph_lib
  gtest_main
  gmock_main
)

add_executable(concatenated_node_test ConcatenatedNodeTest.cc)
target_link_libraries(
  concatenated_node_test
  PRIVATE
  bolt_graph_lib
  gtest_main
  gmock_main
)

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
  gmock_main
)

add_executable(graph_reject_inputs_test GraphRejectsInvalidInputsTest.cc)
target_link_libraries(
  graph_reject_inputs_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

add_executable(state_machine_test NodeStateMachineTest.cc)
target_link_libraries(
  state_machine_test
  PRIVATE
  bolt_graph_lib
  gtest_main
  gmock_main
)

add_executable(embedding_node_test EmbeddingNodeTest.cc)
target_link_libraries(
  embedding_node_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

add_executable(save_load_test SaveLoadTest.cc)
target_link_libraries(
  save_load_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)


include(GoogleTest)
gtest_discover_tests(layer_norm_node_test)
gtest_discover_tests(concatenated_node_test)
gtest_discover_tests(fully_connected_dag_test)
gtest_discover_tests(graph_traversal_test)
gtest_discover_tests(graph_reject_inputs_test)
gtest_discover_tests(state_machine_test)
gtest_discover_tests(embedding_node_test)
gtest_discover_tests(save_load_test)
