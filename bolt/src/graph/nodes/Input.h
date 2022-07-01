#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <sstream>
#include <stdexcept>
#include <string>

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

  void initializeParameters() final {
    if (_expected_input_dim == 0) {
      throw exceptions::GraphCompilationFailure(
          "Cannot have input layer with dimension 0.");
    }
  }

  void forward(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;
    (void)vec_index;
  }

  void backpropagate(uint32_t vec_index) final { (void)vec_index; }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  void setInputs(BoltBatch* inputs) {
    for (uint32_t i = 0; i < inputs->getBatchSize(); i++) {
      checkDimForInput((*inputs)[i]);
    }

    _input_batch = inputs;
  }

  BoltVector& getOutputVector(uint32_t vec_index) final {
    return (*_input_batch)[vec_index];
  }

  uint32_t expectedInputDim() const { return _expected_input_dim; }

  uint32_t outputDim() const final { return _expected_input_dim; }

  uint32_t numNonzerosInOutput() const final {
    throw std::logic_error(
        "Cannot know ahead of time the number of nonzeros "
        "in the output of an Input layer.");
  }

  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity) final {
    (void)batch_size;
    (void)use_sparsity;
  }

  void cleanupAfterBatchProcessing() final {}

  std::vector<NodePtr> getPredecessors() const final { return {}; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    return {};
  }

  bool isInputNode() const final { return true; }

 private:
  void checkDimForInput(const BoltVector& vec) const {
    if (vec.isDense()) {
      if (vec.len != _expected_input_dim) {
        throw std::invalid_argument(
            "Received dense BoltVector with dimension=" +
            std::to_string(vec.len) + " in input layer with dimension=" +
            std::to_string(_expected_input_dim));
      }
    } else {
      for (uint32_t i = 0; i < vec.len; i++) {
        uint32_t active_neuron = vec.active_neurons[i];
        if (active_neuron >= _expected_input_dim) {
          throw std::invalid_argument(
              "Received sparse BoltVector with active_neuron=" +
              std::to_string(active_neuron) +
              " in input layer with dimension=" +
              std::to_string(_expected_input_dim));
        }
      }
    }
  }

  BoltBatch* _input_batch;
  uint32_t _expected_input_dim;
};

}  // namespace thirdai::bolt