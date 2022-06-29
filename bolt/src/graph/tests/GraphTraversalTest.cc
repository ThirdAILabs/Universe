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

  static std::shared_ptr<MockTraversalNode> make(uint32_t id) {
    return std::make_shared<MockTraversalNode>(id);
  }

  uint32_t getID() const { return _id; }

  std::vector<NodePtr> getPredecessors() const final { return _predecessors; }

  void setPredecesors(const std::vector<NodePtr>& predecessors) {
    _predecessors = predecessors;
  }

 private:
  std::vector<NodePtr> _predecessors;
  uint32_t _id;
};

TEST(GraphTraversalTest, CorrectlyTraversesDAG) {
  auto node0 = MockTraversalNode::make(0);
  auto node1 = MockTraversalNode::make(1);
  auto node2 = MockTraversalNode::make(2);
  auto node3 = MockTraversalNode::make(3);
  auto node4 = MockTraversalNode::make(4);
  auto node5 = MockTraversalNode::make(5);
  auto node6 = MockTraversalNode::make(6);

  node6->setPredecesors({node0, node1, node2, node3, node4, node5});
  node5->setPredecesors({node0, node1, node2, node3, node4});
  node4->setPredecesors({node0, node1, node2, node3});
  node3->setPredecesors({node0, node1, node2});
  node2->setPredecesors({node0, node1});
  node1->setPredecesors({node0});

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
  auto node0 = MockTraversalNode::make(0);
  auto node1 = MockTraversalNode::make(1);
  auto node2 = MockTraversalNode::make(2);
  auto node3 = MockTraversalNode::make(3);
  auto node4 = MockTraversalNode::make(4);

  node4->setPredecesors({node3});
  node3->setPredecesors({node2});
  node2->setPredecesors({node1});
  node1->setPredecesors({node0, node3});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node4);

  ASSERT_THROW(graph.compile(std::make_shared<MeanSquaredError>()),  // NOLINT
               exceptions::GraphCompilationFailure);
}

TEST(GraphTraversalTest, ThrowsExceptionForSelfLoop) {
  auto node0 = MockTraversalNode::make(0);
  auto node1 = MockTraversalNode::make(1);

  node1->setPredecesors({node0});
  node0->setPredecesors({node0});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node1);

  ASSERT_THROW(graph.compile(std::make_shared<MeanSquaredError>()),  // NOLINT
               exceptions::GraphCompilationFailure);
}

TEST(GraphTraversalTest, ThrowsExceptionForOutputSelfLoop) {
  auto node0 = MockTraversalNode::make(0);

  node0->setPredecesors({node0});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node0);

  ASSERT_THROW(graph.compile(std::make_shared<MeanSquaredError>()),  // NOLINT
               exceptions::GraphCompilationFailure);
}

}  // namespace thirdai::bolt::tests
