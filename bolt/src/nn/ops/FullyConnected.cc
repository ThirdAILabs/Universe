#include "FullyConnected.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

tensor::ActivationTensorPtr FullyConnected::apply(
    std::shared_ptr<FullyConnectedLayer> kernel, tensor::Tensor* input,
    std::string name, uint32_t rebuild_hash_tables,
    uint32_t reconstruct_hash_functions) {
  auto op = std::shared_ptr<FullyConnected>(
      new FullyConnected(std::move(kernel), input, std::move(name),
                         rebuild_hash_tables, reconstruct_hash_functions));

  input->addDependantOp(op);

  return op->_output;
}

FullyConnected::FullyConnected(std::shared_ptr<FullyConnectedLayer> kernel,
                               tensor::Tensor* input, std::string name,
                               uint32_t rebuild_hash_tables,
                               uint32_t reconstruct_hash_functions)
    : Op(std::move(name)),
      _kernel(std::move(kernel)),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0),
      _input(input) {
  _output = tensor::ActivationTensor::make(_kernel->getDim(),
                                           _kernel->getSparseDim(), this);
}

void FullyConnected::forward(uint32_t index_in_batch) {
  // TODO(Nicholas): Add ability to pass in labels if output layer.
  _kernel->forward(_input->getVector(index_in_batch),
                   _output->getVector(index_in_batch), /* labels= */ nullptr);
}

void FullyConnected::backpropagate(uint32_t index_in_batch) {
  BoltVector& input = _input->getVector(index_in_batch);

  if (input.hasGradients()) {
    _kernel->backpropagate(input, _output->getVector(index_in_batch));
  } else {
    _kernel->backpropagateInputLayer(input, _output->getVector(index_in_batch));
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

void FullyConnected::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

std::vector<tensor::Tensor*> FullyConnected::inputs() const { return {_input}; }

std::vector<tensor::ActivationTensorPtr> FullyConnected::outputs() const {
  return {_output};
}

void FullyConnected::summary(std::ostream& summary) const {
  // TODO(Nicholas) fix this method on the FullyConnectedLayer to take in a
  // std::ostream instead of std::stringstream.
  std::stringstream str_summary;
  _kernel->buildLayerSummary(str_summary, /* detailed= */ true);

  std::string layer_summary = str_summary.str();
  layer_summary.pop_back();  // Get rid of newline

  summary << "FullyConnected(" << name() << "): " << _input->name() << " -> "
          << _output->name() << " (" << layer_summary << ")";
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

std::string nextFullyConnectedOpName() {
  static uint32_t constructed = 0;
  return "fc_" + std::to_string(++constructed);
}

FullyConnectedFactory::FullyConnectedFactory(
    uint32_t dim, float sparsity, std::string activation,
    SamplingConfigPtr sampling, uint32_t rebuild_hash_tables,
    uint32_t reconstruct_hash_functions)
    : _dim(dim),
      _sparsity(sparsity),
      _activation(std::move(activation)),
      _sampling(std::move(sampling)),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _name(nextFullyConnectedOpName()) {
  if (!_sampling) {
    _sampling = DWTASamplingConfig::autotune(_dim, _sparsity);
  }
}

tensor::ActivationTensorPtr FullyConnectedFactory::apply(
    tensor::TensorPtr& input) {
  if (!_kernel) {
    FullyConnectedLayerConfig config(_dim, _sparsity, _activation, _sampling);

    _kernel = std::make_shared<FullyConnectedLayer>(config, input->dim());
  } else if (input->dim() != _kernel->getInputDim()) {
    throw std::invalid_argument(
        "Cannot apply a fully connected layer with input dimension " +
        std::to_string(_kernel->getInputDim()) + " to tensor with dimension " +
        std::to_string(input->dim()) + ".");
  }

  return FullyConnected::apply(_kernel, input.get(), _name,
                               _rebuild_hash_tables,
                               _reconstruct_hash_functions);
}

}  // namespace thirdai::bolt::nn::ops