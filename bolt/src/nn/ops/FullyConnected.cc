#include "FullyConnected.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

tensor::ActivationTensorPtr FullyConnected::apply(
    std::shared_ptr<FullyConnectedLayer> kernel, tensor::TensorPtr input,
    std::string name) {
  auto op = std::make_shared<FullyConnected>(std::move(kernel),
                                             std::move(input), std::move(name));

  input->addDependantOp(op);

  return op->_output;
}

FullyConnected::FullyConnected(std::shared_ptr<FullyConnectedLayer> kernel,
                               tensor::TensorPtr input, std::string name)
    : Op(std::move(name)),
      _kernel(std::move(kernel)),
      _input(std::move(input)) {
  _output = tensor::ActivationTensor::make(_kernel->getDim(),
                                           _kernel->getSparseDim());
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
}

void FullyConnected::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

std::vector<tensor::TensorPtr> FullyConnected::inputs() const {
  return {_input};
}

std::vector<tensor::ActivationTensorPtr> FullyConnected::outputs() const {
  return {_output};
}

std::string nextFullyConnectedOpName() {
  static uint32_t constructed = 0;
  return "fc_" + std::to_string(++constructed);
}

FullyConnectedFactory::FullyConnectedFactory(uint32_t dim, float sparsity,
                                             std::string activation,
                                             SamplingConfigPtr sampling)
    : _dim(dim),
      _sparsity(sparsity),
      _activation(std::move(activation)),
      _sampling(std::move(sampling)),
      _name(nextFullyConnectedOpName()) {
  if (!_sampling) {
    _sampling = DWTASamplingConfig::autotune(_dim, _sparsity);
  }
}

tensor::ActivationTensorPtr FullyConnectedFactory::apply(
    tensor::TensorPtr input) {
  if (!_kernel) {
    FullyConnectedLayerConfig config(_dim, _sparsity, _activation, _sampling);

    _kernel = std::make_shared<FullyConnectedLayer>(config, input->dim());
  } else if (input->dim() != _kernel->getInputDim()) {
    throw std::invalid_argument(
        "Cannot apply a fully connected layer with input dimension " +
        std::to_string(_kernel->getInputDim()) + " to tensor with dimension " +
        std::to_string(input->dim()) + ".");
  }

  return FullyConnected::apply(_kernel, std::move(input), _name);
}

}  // namespace thirdai::bolt::nn::ops