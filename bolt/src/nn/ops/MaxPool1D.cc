#include "MaxPool1D.h"
#include <bolt/src/nn/autograd/Computation.h>
#include <cassert>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

std::string nextMaxPool1DOpName() {
  static uint32_t constructed = 0;
  return "max_pool_1d_" + std::to_string(++constructed);
}

MaxPool1D::MaxPool1D(size_t window_size)
    : Op(nextMaxPool1DOpName()), _window_size(window_size) {}

void MaxPool1D::forward(const ComputationList& inputs, TensorPtr& output,
                        uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const BoltVector& input_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  assert(input_vec.isDense());
  assert(input_vec.len % _window_size == 0);

  BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());
  assert(output_vec.len * _window_size == input_vec.len);

  for (size_t i = 0; i < output_vec.len; i++) {
    size_t start = i * _window_size;
    size_t end = start + _window_size;

    float max = input_vec.activations[start];
    for (size_t j = start + 1; j < end; j++) {
      if (input_vec.activations[j] > max) {
        max = input_vec.activations[j];
      }
    }

    output_vec.activations[i] = max;
  }
}

void MaxPool1D::backpropagate(ComputationList& inputs, TensorPtr& output,
                              uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  BoltVector& input_vec = inputs.at(0)->tensor()->getVector(index_in_batch);
  assert(input_vec.isDense());
  assert(input_vec.len % _window_size == 0);

  if (!input_vec.hasGradients()) {
    return;
  }

  const BoltVector& output_vec = output->getVector(index_in_batch);
  assert(output_vec.isDense());
  assert(output_vec.len * _window_size == input_vec.len);

  for (size_t i = 0; i < output_vec.len; i++) {
    size_t start = i * _window_size;
    size_t end = start + _window_size;

    size_t max_index = start;
    float max = input_vec.activations[start];
    for (size_t j = start + 1; j < end; j++) {
      if (input_vec.activations[j] > max) {
        max = input_vec.activations[j];
        max_index = j;
      }
    }

    input_vec.gradients[max_index] = output_vec.gradients[i];
  }
}

void MaxPool1D::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t MaxPool1D::dim() const { return _output_dim; }

std::optional<uint32_t> MaxPool1D::nonzeros(const ComputationList& inputs,
                                            bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;
  return _output_dim;
}

void MaxPool1D::initOptimizer(const OptimizerFactoryPtr& optimizer_factory) {
  (void)optimizer_factory;
}

void MaxPool1D::disableSparseParameterUpdates() {}

void MaxPool1D::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> MaxPool1D::gradients() { return {}; }

std::vector<std::vector<float>*> MaxPool1D::parameters() { return {}; }

void MaxPool1D::summary(std::ostream& summary, const ComputationList& inputs,
                        const Computation* output) const {
  summary << "MaxPool1D(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << "[window_size=" << _window_size << "]";
}

void MaxPool1D::setSerializeOptimizer(bool should_serialize_optimizer) {
  (void)should_serialize_optimizer;
}

ComputationPtr MaxPool1D::apply(ComputationPtr input) {
  if (_output_dim == 0) {
    _output_dim = input->dim() / _window_size;
  }
  if (input->dim() != _window_size * _output_dim) {
    throw std::invalid_argument(
        "Cannot apply MaxPool1D with window size " +
        std::to_string(_window_size) + " and output dim " +
        std::to_string(_output_dim) + " to input with dim " +
        std::to_string(input->dim()) + ".");
  }

  return Computation::make(shared_from_this(), {std::move(input)});
}

}  // namespace thirdai::bolt