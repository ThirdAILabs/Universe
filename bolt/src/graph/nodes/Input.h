#pragma once

#include <cereal/types/polymorphic.hpp>
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
      : _input_batch(nullptr), _expected_input_dim(expected_input_dim) {}

  void setInputs(BoltBatch* inputs) {
    for (uint32_t i = 0; i < inputs->getBatchSize(); i++) {
      checkDimForInput((*inputs)[i]);
    }

    _input_batch = inputs;
  }

  uint32_t expectedInputDim() const { return _expected_input_dim; }

  uint32_t outputDim() const final { return _expected_input_dim; }

  uint32_t numNonzerosInOutput() const final {
    throw std::logic_error(
        "Cannot know ahead of time the number of nonzeros "
        "in the output of an Input layer.");
  }

  std::vector<NodePtr> getPredecessors() const final { return {}; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    return {};
  }

  bool isInputNode() const final { return true; }

 private:
  void initializeParametersImpl() final {
    if (_expected_input_dim == 0) {
      throw exceptions::GraphCompilationFailure(
          "Cannot have input layer with dimension 0.");
    }
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)batch_size;
    (void)use_sparsity;

    if (preparedForBatchProcessing()) {
      throw exceptions::NodeStateMachineError(
          "Input should have setBatch called before "
          "prepareForBatchProcessing.");
    }
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;
    (void)vec_index;
  }

  void backpropagateImpl(uint32_t vec_index) final { (void)vec_index; }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    assert(preparedForBatchProcessing());

    return (*_input_batch)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final {}

  bool predecessorsSet() const final { return true; }

  bool parametersInitialized() const final { return true; }

  bool preparedForBatchProcessing() const final {
    return _input_batch != nullptr;
  }

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

  // Private constructor for cereal.
  Input() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _expected_input_dim);
  }

  BoltBatch* _input_batch;
  uint32_t _expected_input_dim;
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Input)