#include "FullyConnected.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextFullyConnectedOpName() {
  static uint32_t constructed = 0;
  return "fc_" + std::to_string(++constructed);
}

std::shared_ptr<FullyConnected> FullyConnected::make(
    uint32_t dim, uint32_t input_dim, float sparsity,
    const std::string& activation, SamplingConfigPtr sampling,
    uint32_t rebuild_hash_tables, uint32_t reconstruct_hash_functions) {
  return std::shared_ptr<FullyConnected>(new FullyConnected(
      dim, input_dim, sparsity, activation, std::move(sampling),
      rebuild_hash_tables, reconstruct_hash_functions));
}

FullyConnected::FullyConnected(uint32_t dim, uint32_t input_dim, float sparsity,
                               const std::string& activation,
                               SamplingConfigPtr sampling,
                               uint32_t rebuild_hash_tables,
                               uint32_t reconstruct_hash_functions)
    : Op(nextFullyConnectedOpName()),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {
  if (!sampling) {
    sampling = DWTASamplingConfig::autotune(dim, sparsity);
  }
  FullyConnectedLayerConfig config(dim, sparsity, activation,
                                   std::move(sampling));

  _kernel = std::make_shared<FullyConnectedLayer>(config, input_dim);
}

void FullyConnected::forward(const tensor::TensorList& inputs,
                             tensor::ActivationTensor* output,
                             uint32_t index_in_batch, bool training) {
  // If the op is an output pass in labels during training to ensure labels are
  // in active neuron set.
  const BoltVector* labels = nullptr;
  (void)training;
  if (training && inputs.size() == 2) {
    labels = &inputs[1]->getVector(index_in_batch);
  }
  _kernel->forward(inputs[0]->getVector(index_in_batch),
                   output->getVector(index_in_batch), labels);
}

void FullyConnected::backpropagate(tensor::TensorList& inputs,
                                   tensor::ActivationTensor* output,
                                   uint32_t index_in_batch) {
  BoltVector& input = inputs[0]->getVector(index_in_batch);

  if (input.hasGradients()) {
    _kernel->backpropagate(input, output->getVector(index_in_batch));
  } else {
    _kernel->backpropagateInputLayer(input, output->getVector(index_in_batch));
  }
}

void FullyConnected::updateParameters(float learning_rate,
                                      uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);

  if (++_updates_since_reconstruct_hash_functions ==
      _reconstruct_hash_functions) {
    _kernel->reBuildHashFunction();
    _kernel->buildHashTables();

    _updates_since_rebuild_hash_tables = 0;
    _updates_since_reconstruct_hash_functions = 0;
  } else if (++_updates_since_rebuild_hash_tables == _rebuild_hash_tables) {
    _kernel->buildHashTables();
    _updates_since_rebuild_hash_tables = 0;
  }
}

uint32_t FullyConnected::numNonzerosInOutput(const tensor::TensorList& inputs,
                                             bool use_sparsity) const {
  // The number of output nonzeros for a FullyConnected op do not depend on its
  // inputs.
  (void)inputs;
  if (use_sparsity) {
    return _kernel->getSparseDim();
  }
  return _kernel->getDim();
}

void FullyConnected::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

void FullyConnected::summary(std::ostream& summary,
                             const tensor::TensorList& inputs,
                             const tensor::ActivationTensor* output) const {
  summary << "FullyConnected(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
  summary << " :: dim=" << _kernel->getDim()
          << ", sparsity=" << _kernel->getSparsity() << ", activation="
          << activationFunctionToStr(_kernel->getActivationFunction());
  if (_kernel->getSparsity() < 1.0) {
    summary << ", sampling=(";
    _kernel->buildSamplingSummary(summary);
    summary << ", rebuild_hash_tables=" << _rebuild_hash_tables;
    summary << ", reconstruct_hash_functions=" << _reconstruct_hash_functions;
    summary << ")";
  }
}

tensor::ActivationTensorPtr FullyConnected::apply(tensor::TensorPtr input) {
  if (input->dim() != _kernel->getInputDim()) {
    std::stringstream error;
    error << "Cannot apply FullyConnected op with weight matrix of shape ("
          << _kernel->getDim() << ", " << _kernel->getInputDim()
          << ") to input tensor with dim " << input->dim() << ".";

    throw std::invalid_argument(error.str());
  }
  return tensor::ActivationTensor::make(_kernel->getDim(), shared_from_this(),
                                        {std::move(input)});
}

std::vector<uint32_t> FullyConnected::dimensions() const {
  return {_kernel->getDim(), _kernel->getInputDim()};
}

const float* FullyConnected::weightsPtr() const {
  return _kernel->getWeightsPtr();
}

const float* FullyConnected::biasesPtr() const {
  return _kernel->getBiasesPtr();
}

}  // namespace thirdai::bolt::nn::ops