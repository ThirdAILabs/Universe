

#include "MockNode.h"
#include <bolt/src/graph/nodes/Concatenated.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

namespace thirdai::bolt::tests {

// state 1 = predecessors not set
// state 2 = predecessors set
// state 3 = prepared for batch processing

// TODO(Josh): Make this test general to other nodes once we add state machines
// to them.
// TODO(Josh): Do we want to try to test forward and backpropogate and other
//  methods that use assertions instead of exceptions?

void testBadCallsInState1(
    const std::shared_ptr<ConcatenatedNode>& concatenated_node) {
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      concatenated_node->prepareForBatchProcessing(/* batch_size = */ 0,
                                                   /* use_sparsity = */ false),
      exceptions::NodeStateMachineError);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      concatenated_node->cleanupAfterBatchProcessing(),
      exceptions::NodeStateMachineError);
}

void testBadCallsInState2(
    const std::shared_ptr<ConcatenatedNode>& concatenated_node,
    const NodePtr& mock_input) {
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      concatenated_node->setConcatenatedNodes({mock_input, mock_input}),
      exceptions::NodeStateMachineError);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      concatenated_node->cleanupAfterBatchProcessing(),
      exceptions::NodeStateMachineError);
}

void testBadCallsInState3(
    const std::shared_ptr<ConcatenatedNode>& concatenated_node,
    const NodePtr& mock_input) {
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      concatenated_node->setConcatenatedNodes({mock_input, mock_input}),
      exceptions::NodeStateMachineError);
  ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
      concatenated_node->prepareForBatchProcessing(/* batch_size = */ 0,
                                                   /* use_sparsity = */ false),
      exceptions::NodeStateMachineError);
}

void moveNodeState1ToState2(
    const std::shared_ptr<ConcatenatedNode>& concatenated_node,
    const NodePtr& mock_input) {
  concatenated_node->setConcatenatedNodes({mock_input, mock_input});
}

void moveNodeState2ToState3(
    const std::shared_ptr<ConcatenatedNode>& concatenated_node) {
  concatenated_node->prepareForBatchProcessing(/* batch_size = */ 0,
                                               /* use_sparsity = */ false);
}

void moveNodeState3ToState2(
    const std::shared_ptr<ConcatenatedNode>& concatenated_node) {
  concatenated_node->cleanupAfterBatchProcessing();
}

TEST(NodeStateMachineTest, ConcatenateStateMachine) {
  BoltVector node_1_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {0.5, 0.75});
  NodePtr mock_input = std::make_shared<MockNodeWithOutput>(
      node_1_output, /* output_dense_dim = */ 2);
  auto concatenated_node = std::make_shared<ConcatenatedNode>();

  // Node now in state 1

  testBadCallsInState1(concatenated_node);

  moveNodeState1ToState2(concatenated_node, mock_input);

  testBadCallsInState2(concatenated_node, mock_input);

  uint32_t cycle_iterations = 10;
  for (uint32_t i = 0; i < cycle_iterations; i++) {
    moveNodeState2ToState3(concatenated_node);

    testBadCallsInState3(concatenated_node, mock_input);

    moveNodeState3ToState2(concatenated_node);

    testBadCallsInState2(concatenated_node, mock_input);
  }
}

}  // namespace thirdai::bolt::tests
