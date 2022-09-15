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

add_executable(switch_node_test SwitchNodeTest.cc)
target_link_libraries(
  switch_node_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

add_executable(dag_dlrm_test DlrmTests.cc)
target_link_libraries(
  dag_dlrm_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

add_executable(predict_single_test PredictSingleTest.cc)
target_link_libraries(
  predict_single_test
  PRIVATE
  bolt_graph_lib
  gtest_main
)

include(GoogleTest)

set(DAG_TESTS
  layer_norm_node_test
  concatenated_node_test
  fully_connected_dag_test
  graph_traversal_test
  graph_reject_inputs_test
  state_machine_test
  embedding_node_test
  save_load_test
  switch_node_test
  dag_dlrm_test
  predict_single_test
)

foreach(DAG_TEST IN LISTS DAG_TESTS)
  gtest_discover_tests(${DAG_TEST} DISCOVERY_TIMEOUT ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
