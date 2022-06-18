#pragma once

#include <bolt/src/graph/Node.h>
#include <stdexcept>

namespace thirdai::bolt {

// A node subclass for input layers. The input batch will be stored in this
// layer so that subsequent layers can access the inputs through its getOutput()
// method. This makes the interface simplier by generalizing the forward pass so
// that other layers always just access the outputs of the previous layer rather
// than have to worry if they they need to access an input directly or access
// the outputs of a previous layer.
class Input final : public Node {
 public:
  explicit Input(uint32_t expected_input_dim)
      : _expected_input_dim(expected_input_dim) {}

  void compile() final {}

  void forward(uint32_t batch_index, const BoltVector* labels) final {
    (void)labels;
    (void)batch_index;
  }

  void backpropagate(uint32_t batch_index) final { (void)batch_index; }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  void setInputs(BoltBatch* inputs) { _input_batch = inputs; }

  BoltVector& getOutput(uint32_t batch_index) final {
    return (*_input_batch)[batch_index];
  }

  uint32_t expectedInputDim() const { return _expected_input_dim; }

  uint32_t outputDim() const final { return _expected_input_dim; }

  bool hasSparseOutput() const final {
    throw std::logic_error(
        "Cannot determine if Input is sparse or dense until runtime");
  }

  uint32_t sparseOutputDim() const final {
    throw std::logic_error(
        "Cannot access sparseOutputDim of input layer since the number of "
        "nonzeros cannot be known until runtime");
  }

  void initializeState(uint32_t batch_size, bool is_inference) final {
    (void)batch_size;
    (void)is_inference;
  }

  void enqueuePredecessors(std::queue<NodePtr>& nodes) final { (void)nodes; }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    (void)sparse_layers;
  }

  bool isInputNode() const final { return true; }

 private:
  BoltBatch* _input_batch;
  uint32_t _expected_input_dim;
};

}  // namespace thirdai::bolt