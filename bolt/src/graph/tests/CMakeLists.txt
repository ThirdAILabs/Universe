enable_testing()

include(GoogleTest)

set(DAG_MOCK_TESTS LayerNormNodeTest ConcatenatedNodeTest GraphTraversalTest
                   NodeStateMachineTest)

set(DAG_MAIN_TESTS
    FullyConnectedDagTest
    GraphRejectsInvalidInputsTest
    EmbeddingNodeTest
    SaveLoadTest
    SwitchNodeTest
    DlrmTests
    DlrmAttentionNodeTests
    PredictSingleTest)

foreach(DAG_MOCK_TEST IN LISTS DAG_MOCK_TESTS)
  add_executable(${DAG_MOCK_TEST} ${DAG_MOCK_TEST}.cc)
  target_link_libraries(${DAG_MOCK_TEST} PRIVATE thirdai gtest_main gmock_main)
  gtest_discover_tests(${DAG_MOCK_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()

foreach(DAG_MAIN_TEST IN LISTS DAG_MAIN_TESTS)
  add_executable(${DAG_MAIN_TEST} ${DAG_MAIN_TEST}.cc)
  target_link_libraries(${DAG_MAIN_TEST} PRIVATE thirdai gtest_main)
  gtest_discover_tests(${DAG_MAIN_TEST} DISCOVERY_TIMEOUT
                       ${THIRDAI_GTEST_DISCOVERY_TIMEOUT})
endforeach()
