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

  MOCK_METHOD(void, prepareForBatchProcessing,
              (uint32_t batch_size, bool use_sparsity), (override));

  MOCK_METHOD(std::vector<NodePtr>, getPredecessors, (), (const override));

  MOCK_METHOD(std::vector<std::shared_ptr<FullyConnectedLayer>>,
              getInternalFullyConnectedLayers, (), (const override));

  MOCK_METHOD(bool, isInputNode, (), (const override));
};

}  // namespace thirdai::bolt::tests