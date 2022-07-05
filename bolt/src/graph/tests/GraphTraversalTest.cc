#include "MockNode.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <memory>
#include <unordered_set>

namespace thirdai::bolt::tests {

class MockTraversalNode final : public MockNode {
 public:
  explicit MockTraversalNode(uint32_t id) : _id(id) {}

  static std::shared_ptr<MockTraversalNode> makePtr(uint32_t id) {
    return std::make_shared<MockTraversalNode>(id);
  }

  uint32_t getID() const { return _id; }

  void setPredecessors(const std::vector<NodePtr>& predecessors) {
    _predecessors = predecessors;
    _state = NodeState::PredecessorsSet;
  }

 private:
  std::vector<NodePtr> getPredecessorsImpl() const final {
    return _predecessors;
  }

  void compileImpl() final { _state = NodeState::Compiled; }

  const std::string& type() const final { return _type; }

  NodeState getState() const final { return _state; }

  std::vector<NodePtr> _predecessors;
  uint32_t _id;
  NodeState _state = NodeState::Constructed;
  std::string _type = "test";
};

TEST(GraphTraversalTest, CorrectlyTraversesDAG) {
  auto node0 = MockTraversalNode::makePtr(0);
  auto node1 = MockTraversalNode::makePtr(1);
  auto node2 = MockTraversalNode::makePtr(2);
  auto node3 = MockTraversalNode::makePtr(3);
  auto node4 = MockTraversalNode::makePtr(4);
  auto node5 = MockTraversalNode::makePtr(5);
  auto node6 = MockTraversalNode::makePtr(6);

  node6->setPredecessors({node0, node1, node2, node3, node4, node5});
  node5->setPredecessors({node0, node1, node2, node3, node4});
  node4->setPredecessors({node0, node1, node2, node3});
  node3->setPredecessors({node0, node1, node2});
  node2->setPredecessors({node0, node1});
  node1->setPredecessors({node0});
  node0->setPredecessors({});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node6);

  graph.compile(std::make_shared<MeanSquaredError>());

  std::vector<NodePtr> graph_nodes = graph.getNodeTraversalOrder();

  ASSERT_EQ(graph_nodes.size(), 7);

  for (uint32_t node_indx = 0; node_indx < graph_nodes.size(); node_indx++) {
    MockTraversalNode* dummy_node =
        dynamic_cast<MockTraversalNode*>(graph_nodes[node_indx].get());
    ASSERT_NE(dummy_node, nullptr);
    ASSERT_EQ(dummy_node->getID(), node_indx);
  }
}

TEST(GraphTraversalTest, ThrowsExceptionForCycle) {
  auto node0 = MockTraversalNode::makePtr(0);
  auto node1 = MockTraversalNode::makePtr(1);
  auto node2 = MockTraversalNode::makePtr(2);
  auto node3 = MockTraversalNode::makePtr(3);
  auto node4 = MockTraversalNode::makePtr(4);

  node4->setPredecessors({node3});
  node3->setPredecessors({node2});
  node2->setPredecessors({node1});
  node1->setPredecessors({node0, node3});
  node0->setPredecessors({});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node4);

  ASSERT_THROW(graph.compile(std::make_shared<MeanSquaredError>()),  // NOLINT
               exceptions::GraphCompilationFailure);

  // We have to break the cycle of smart pointers to avoid a memory leak here.
  node3->setPredecessors({});
}

TEST(GraphTraversalTest, ThrowsExceptionForSelfLoop) {
  auto node0 = MockTraversalNode::makePtr(0);
  auto node1 = MockTraversalNode::makePtr(1);

  node1->setPredecessors({node0});
  node0->setPredecessors({node0});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node1);

  ASSERT_THROW(graph.compile(std::make_shared<MeanSquaredError>()),  // NOLINT
               exceptions::GraphCompilationFailure);

  // We have to break the cycle of smart pointers to avoid a memory leak here.
  node0->setPredecessors({});
}

TEST(GraphTraversalTest, ThrowsExceptionForOutputSelfLoop) {
  auto node0 = MockTraversalNode::makePtr(0);

  node0->setPredecessors({node0});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node0);

  ASSERT_THROW(graph.compile(std::make_shared<MeanSquaredError>()),  // NOLINT
               exceptions::GraphCompilationFailure);

  // We have to break the cycle of smart pointers to avoid a memory leak here.
  node0->setPredecessors({});
}

}  // namespace thirdai::bolt::tests
