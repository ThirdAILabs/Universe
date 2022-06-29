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

TEST(GraphTraversalTest, CorrectlyTraversesDAG) {
  auto node0 = DummyNode::makeDummyNode(0);
  auto node1 = DummyNode::makeDummyNode(1);
  auto node2 = DummyNode::makeDummyNode(2);
  auto node3 = DummyNode::makeDummyNode(3);
  auto node4 = DummyNode::makeDummyNode(4);
  auto node5 = DummyNode::makeDummyNode(5);
  auto node6 = DummyNode::makeDummyNode(6);

  node6->setPredecesors({node0, node1, node2, node3, node4, node5});
  node5->setPredecesors({node0, node1, node2, node3, node4});
  node4->setPredecesors({node0, node1, node2, node3});
  node3->setPredecesors({node0, node1, node2});
  node2->setPredecesors({node0, node1});
  node1->setPredecesors({node0});

  BoltGraph graph(/* inputs= */ {}, /* output= */ node6);

  graph.compile(std::make_shared<MeanSquaredError>());

  std::vector<NodePtr> graph_nodes = graph.getNodeTraversalOrder();

  ASSERT_EQ(graph_nodes.size(), 6);

  for (uint32_t node_indx = 0; node_indx < graph_nodes.size(); node_indx++) {
    DummyNode* dummy_node =
        dynamic_cast<DummyNode*>(graph_nodes[node_indx].get());
    ASSERT_NE(dummy_node, nullptr);
    ASSERT_EQ(dummy_node->getID(), node_indx + 1);
  }
}

}  // namespace thirdai::bolt::tests
