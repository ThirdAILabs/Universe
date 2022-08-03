#pragma once

#include "gmock/gmock.h"
#include <bolt/src/graph/Node.h>

namespace thirdai::bolt::tests {

// See https://google.github.io/googletest/gmock_for_dummies.html
class MockNode : public Node {
 public:
  MOCK_METHOD(uint32_t, outputDim, (), (const override));

  MOCK_METHOD(uint32_t, numNonzerosInOutput, (), (const override));

  MOCK_METHOD(std::vector<NodePtr>, getPredecessors, (), (const override));

  MOCK_METHOD(bool, isInputNode, (), (const override));

  MOCK_METHOD(void, isDistributedTraining, (), (override));

 private:
  MOCK_METHOD(void, compileImpl, (), (override));

  MOCK_METHOD(std::vector<std::shared_ptr<FullyConnectedLayer>>,
              getInternalFullyConnectedLayersImpl, (), (const override));

  MOCK_METHOD(void, prepareForBatchProcessingImpl,
              (uint32_t batch_size, bool use_sparsity), (override));

  MOCK_METHOD(uint32_t, numNonzerosInOutputImpl, (), (const override));

  MOCK_METHOD(void, forwardImpl, (uint32_t vec_index, const BoltVector* labels),
              (override));

  MOCK_METHOD(void, backpropagateImpl, (uint32_t vec_index), (override));

  MOCK_METHOD(void, updateParametersImpl,
              (float learning_rate, uint32_t batch_cnt), (override));

  MOCK_METHOD(BoltVector&, getOutputVectorImpl, (uint32_t vec_index),
              (override));

  MOCK_METHOD(void, cleanupAfterBatchProcessingImpl, (), (override));

  MOCK_METHOD(void, summarizeImpl, (std::stringstream & summary, bool detailed),
              (const override));

  MOCK_METHOD(std::string, type, (), (const override));

  MOCK_METHOD(std::vector<NodePtr>, getPredecessorsImpl, (), (const override));

  MOCK_METHOD(const std::string&, nameImpl, (), (const ovveride));

  MOCK_METHOD(NodeState, getState, (), (const ovveride));
};

class MockNodeWithOutput : public MockNode {
 public:
  explicit MockNodeWithOutput(BoltVector output, uint32_t output_dense_dim)
      : _output(std::move(output)), _output_dim(output_dense_dim) {}

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    (void)vec_index;
    return _output;
  }

  uint32_t outputDim() const final { return _output_dim; }

  void isDistributedTraining() final {}

  NodeState getState() const final {
    return NodeState::PreparedForBatchProcessing;
  }

 private:
  uint32_t numNonzerosInOutputImpl() const final { return _output.len; }

  BoltVector _output;
  uint32_t _output_dim;
};

}  // namespace thirdai::bolt::tests