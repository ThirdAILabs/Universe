#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <memory>
#include <unordered_set>

namespace thirdai::bolt::tests {

class DummyNode final : public Node {
 public:
  explicit DummyNode(uint32_t id) : _id(id) {}

  static std::shared_ptr<DummyNode> makeDummyNode(uint32_t id) {
    return std::make_shared<DummyNode>(id);
  }

  // These are the only methods that will be called in this node subclass.
  std::vector<NodePtr> getPredecessors() const final { return _predecessors; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    return {};
  }

  bool isInputNode() const final { return false; }

  void setPredecesors(const std::vector<NodePtr>& predecessors) {
    _predecessors = predecessors;
  }

  uint32_t getID() const { return _id; }

  void initializeParameters() final {}

  // These remaining methods are required for the interface but are not used for
  // this test.
  void forward(uint32_t vec_index, const BoltVector* labels) final {
    (void)vec_index;
    (void)labels;
    throw exceptions::NotImplemented("Dummy method for test");
  }

  void backpropagate(uint32_t vec_index) final { (void)vec_index; }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
    throw exceptions::NotImplemented("Dummy method for test");
  }

  BoltVector& getOutputVector(uint32_t vec_index) final {
    (void)vec_index;
    throw exceptions::NotImplemented("Dummy method for test");
  }

  uint32_t outputDim() const final {
    throw exceptions::NotImplemented("Dummy method for test");
  }

  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity) final {
    (void)batch_size;
    (void)use_sparsity;
    throw exceptions::NotImplemented("Dummy method for test");
  }

 private:
  std::vector<NodePtr> _predecessors;
  uint32_t _id;
};

class GraphTraversalTestFixture : public ::testing::Test {
 public:
  static std::vector<NodePtr> getNodes(BoltGraph& graph) {
    return graph._nodes;
  }
};

TEST_F(GraphTraversalTestFixture, CorrectlyTraversesDAG) {
  auto node1 = DummyNode::makeDummyNode(1);
  auto node2 = DummyNode::makeDummyNode(2);
  auto node3 = DummyNode::makeDummyNode(3);
  auto node4 = DummyNode::makeDummyNode(4);
  auto node5 = DummyNode::makeDummyNode(5);
  auto node6 = DummyNode::makeDummyNode(6);
  auto node7 = DummyNode::makeDummyNode(7);
  auto node8 = DummyNode::makeDummyNode(8);

  node4->setPredecesors({node1, node2});
  node5->setPredecesors({node3});

  node6->setPredecesors({node5});
  node7->setPredecesors({node5});

  node8->setPredecesors({node4, node6, node7});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node8);

  graph.compile(std::make_shared<MeanSquaredError>());

  std::vector<uint32_t> expected_ids = {8, 4, 6, 7, 1, 2, 5, 3};
  std::reverse(expected_ids.begin(), expected_ids.end());

  std::vector<NodePtr> graph_nodes = getNodes(graph);

  ASSERT_EQ(graph_nodes.size(), expected_ids.size());

  for (uint32_t node_indx = 0; node_indx < graph_nodes.size(); node_indx++) {
    DummyNode* dummy_node =
        dynamic_cast<DummyNode*>(graph_nodes[node_indx].get());
    ASSERT_NE(dummy_node, nullptr);
    ASSERT_EQ(dummy_node->getID(), expected_ids[node_indx]);
  }
}

}  // namespace thirdai::bolt::tests
