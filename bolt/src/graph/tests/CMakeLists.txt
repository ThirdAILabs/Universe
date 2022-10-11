enable_testing()

add_executable(layer_norm_node_test LayerNormNodeTest.cc)

add_executable(concatenated_node_test ConcatenatedNodeTest.cc)

add_executable(fully_connected_dag_test FullyConnectedDagTest.cc)

add_executable(graph_traversal_test GraphTraversalTest.cc)

add_executable(graph_reject_inputs_test GraphRejectsInvalidInputsTest.cc)

add_executable(state_machine_test NodeStateMachineTest.cc)

add_executable(embedding_node_test EmbeddingNodeTest.cc)

add_executable(save_load_test SaveLoadTest.cc)

add_executable(switch_node_test SwitchNodeTest.cc)

add_executable(dag_dlrm_test DlrmTests.cc)

add_executable(dag_dlrm_attention_test DlrmAttentionNodeTests.cc)

add_executable(predict_single_test PredictSingleTest.cc)

include(GoogleTest)

set(DAG_MOCK_TESTS layer_norm_node_test concatenated_node_test
                   graph_traversal_test state_machine_test)

set(DAG_MAIN_TESTS
    fully_connected_dag_test
    graph_reject_inputs_test
    embedding_node_test
    save_load_test
    switch_node_test
    dag_dlrm_test
    dag_dlrm_attention_test
    predict_single_test)

foreach(DAG_MOCK_TEST IN LISTS DAG_MOCK_TESTS)
  target_link_libraries(${DAG_MOCK_TEST} PRIVATE thirdai gtest_main gmock_main)
  gtest_discover_tests(${DAG_MOCK_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()

foreach(DAG_MAIN_TEST IN LISTS DAG_MAIN_TESTS)
  target_link_libraries(${DAG_MAIN_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${DAG_MAIN_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
