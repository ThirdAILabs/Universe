#pragma once

#include "gmock/gmock.h"
#include <bolt/src/graph/Node.h>

namespace thirdai::bolt::tests {

// See https://google.github.io/googletest/gmock_for_dummies.html
class MockNode : public Node {
 public:
  MOCK_METHOD(void, initializeParameters, (), (override));

  MOCK_METHOD(void, forward, (uint32_t vec_index, const BoltVector* labels),
              (override));

  MOCK_METHOD(void, backpropagate, (uint32_t vec_index), (override));

  MOCK_METHOD(void, updateParameters, (float learning_rate, uint32_t batch_cnt),
              (override));

  MOCK_METHOD(BoltVector&, getOutputVector, (uint32_t vec_index), (override));

  MOCK_METHOD(uint32_t, outputDim, (), (const override));

  MOCK_METHOD(uint32_t, numNonzerosInOutput, (), (const override));

  MOCK_METHOD(void, prepareForBatchProcessing,
              (uint32_t batch_size, bool use_sparsity), (override));

  MOCK_METHOD(void, cleanupAfterBatchProcessing, (), (override));

  MOCK_METHOD(std::vector<NodePtr>, getPredecessors, (), (const override));

  MOCK_METHOD(std::vector<std::shared_ptr<FullyConnectedLayer>>,
              getInternalFullyConnectedLayers, (), (const override));

  MOCK_METHOD(bool, isInputNode, (), (const override));

  MOCK_METHOD(void, summarize, (std::stringstream & summary, bool detailed),
              (const override));

  MOCK_METHOD(
      void, setNameAndUpdateCount,
      ((std::unordered_map<std::string, uint32_t>)&layer_type_name_to_count),
      (override));

  MOCK_METHOD(const std::string&, name, (), (const ovveride));
};

class MockNodeWithOutput : public MockNode {
 public:
  explicit MockNodeWithOutput(BoltVector output, uint32_t output_dense_dim)
      : _output(std::move(output)), _output_dim(output_dense_dim) {}

  BoltVector& getOutputVector(uint32_t vec_index) final {
    (void)vec_index;
    return _output;
  }

  uint32_t outputDim() const final { return _output_dim; }

  uint32_t numNonzerosInOutput() const final { return _output.len; }

 private:
  BoltVector _output;
  uint32_t _output_dim;
};

}  // namespace thirdai::bolt::tests