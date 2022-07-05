

#include "MockNode.h"
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerUtils.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

namespace thirdai::bolt::tests {

// state 1 = predecessors not set
// state 2 = predecessors set
// state 3 = parameters initialized
// state 4 = prepared for batch processing

// TODO(Josh): Do we want to try to test forward and backpropogate and other
//  methods that use assertions instead of exceptions?

class NodeStateMachineTest {
 public:
  explicit NodeStateMachineTest(NodePtr node) : _node(std::move(node)) {
    _mock_output =
        BoltVector::makeDenseVectorWithGradients(/* values = */ {0.5, 0.75});
    _mock_node = std::make_shared<MockNodeWithOutput>(
        _mock_output, /* output_dense_dim = */ 2);
  }

  void runTest() {
    testBadCallsInState1();

    moveNodeState1ToState2();
    testBadCallsInState2();

    moveNodeState2ToState3();
    testBadCallsInState3();

    uint32_t cycle_iterations = 10;
    for (uint32_t i = 0; i < cycle_iterations; i++) {
      moveNodeState3ToState4();
      testBadCallsInState4();

      moveNodeState4ToState3();
      testBadCallsInState3();
    }
  }

 private:
  virtual void setNodePrecessors(const std::vector<NodePtr>& predecessors) = 0;

  // Methods for moving between states.
  void moveNodeState1ToState2() {
    ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
        setNodePrecessors({_mock_node, _mock_node}));
  }

  void moveNodeState2ToState3() {
    ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
        _node->initializeParameters());
  }

  void moveNodeState3ToState4() {
    ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
        _node->prepareForBatchProcessing(
            /* batch_size = */ 0,
            /* use_sparsity = */ false));
  }

  void moveNodeState4ToState3() {
    ASSERT_NO_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_NO_THROW
        _node->cleanupAfterBatchProcessing());
  }

  // Methods for testing invalid calls within each state.
  void testBadCallsInState1() {
    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->initializeParameters(), exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->prepareForBatchProcessing(
            /* batch_size = */ 0,
            /* use_sparsity = */ false),
        exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->cleanupAfterBatchProcessing(),
        exceptions::NodeStateMachineError);
  }

  void testBadCallsInState2() {
    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        setNodePrecessors({_mock_node}), exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->prepareForBatchProcessing(
            /* batch_size = */ 0,
            /* use_sparsity = */ false),
        exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->cleanupAfterBatchProcessing(),
        exceptions::NodeStateMachineError);
  }

  void testBadCallsInState3() {
    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        setNodePrecessors({_mock_node}), exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->initializeParameters(), exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->cleanupAfterBatchProcessing(),
        exceptions::NodeStateMachineError);
  }

  void testBadCallsInState4() {
    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        setNodePrecessors({_mock_node}), exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->initializeParameters(), exceptions::NodeStateMachineError);

    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        _node->prepareForBatchProcessing(
            /* batch_size = */ 0,
            /* use_sparsity = */ false),
        exceptions::NodeStateMachineError);
  }

  BoltVector _mock_output;
  NodePtr _mock_node;

 protected:
  NodePtr _node;
};

class ConcatenateStateMachineTest final : public NodeStateMachineTest {
 public:
  explicit ConcatenateStateMachineTest(NodePtr node)
      : NodeStateMachineTest(std::move(node)) {}

 private:
  void setNodePrecessors(const std::vector<NodePtr>& predecessors) override {
    ConcatenateNode* concat_node = dynamic_cast<ConcatenateNode*>(_node.get());
    ASSERT_NE(concat_node, nullptr);

    concat_node->setConcatenatedNodes(predecessors);
  }
};

TEST(NodeStateMachineTest, ConcatenateStateMachine) {
  auto concatenated_node = std::make_shared<ConcatenateNode>();

  ConcatenateStateMachineTest test(concatenated_node);

  test.runTest();
}

class FullyConnectedStateMachineTest final : public NodeStateMachineTest {
 public:
  explicit FullyConnectedStateMachineTest(NodePtr node)
      : NodeStateMachineTest(std::move(node)) {}

 private:
  void setNodePrecessors(const std::vector<NodePtr>& predecessors) override {
    FullyConnectedNode* fully_connected =
        dynamic_cast<FullyConnectedNode*>(_node.get());
    ASSERT_NE(fully_connected, nullptr);

    fully_connected->addPredecessor(predecessors.at(0));
  }
};

TEST(NodeStateMachineTest, FullyConnectedStateMachine) {
  auto fully_connected_node = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10, /* activation= */ ActivationFunction::ReLU);

  FullyConnectedStateMachineTest test(fully_connected_node);

  test.runTest();
}

class EmbeddingStateMachineTest final : public NodeStateMachineTest {
 public:
  explicit EmbeddingStateMachineTest(NodePtr node)
      : NodeStateMachineTest(std::move(node)) {}

 private:
  void setNodePrecessors(const std::vector<NodePtr>& predecessors) override {
    (void)predecessors;

    EmbeddingNode* embedding_node = dynamic_cast<EmbeddingNode*>(_node.get());
    ASSERT_NE(embedding_node, nullptr);

    TokenInputPtr input = std::make_shared<TokenInput>();

    embedding_node->addInput(input);
  }
};

TEST(NodeStateMachineTest, EmbeddingStateMachine) {
  auto embedding_node = std::make_shared<EmbeddingNode>(
      /* num_embedding_lookups= */ 10, /* lookup_size= */ 8,
      /* log_embedding_block_size= */ 5);

  EmbeddingStateMachineTest test(embedding_node);

  test.runTest();
}

}  // namespace thirdai::bolt::tests
