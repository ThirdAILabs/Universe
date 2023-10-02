#include "Sparsify.h"
#include <bolt/src/nn/autograd/Computation.h>
#include <algorithm>
#include <stdexcept>

namespace thirdai::bolt {

void Sparsify::forward(const ComputationList& inputs, TensorPtr& output,
                       uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const BoltVector& input_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  if (!input_vec.isDense()) {
    throw std::invalid_argument("Sparsify expects dense input.");
  }

  BoltVector& output_vec = output->getVector(index_in_batch);

  if (output_vec.isDense()) {
    assert(input_vec.isDense());
    assert(input_vec.len == output_vec.len);
    
    std::copy(input_vec.activations, input_vec.activations + input_vec.len,
              output_vec.activations);
  } else {
    auto top_k = input_vec.findKLargestActivations(output_vec.len);

    size_t i = 0;
    while (!top_k.empty()) {
      output_vec.active_neurons[i] = top_k.top().second;
      output_vec.activations[i] = top_k.top().first;
      top_k.pop();
      i++;
    }
  }
}

void Sparsify::backpropagate(ComputationList& inputs, TensorPtr& output,
                             uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  const BoltVector& input_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);

  BoltVector& output_vec = output->getVector(index_in_batch);

  if (output_vec.isDense()) {
    assert(input_vec.isDense());
    assert(input_vec.len == output_vec.len);

    std::copy(output_vec.gradients, output_vec.gradients + output_vec.len,
              input_vec.gradients);
  } else {
    for (size_t i = 0; i < output_vec.len; i++) {
      input_vec.gradients[output_vec.active_neurons[i]] +=
          output_vec.gradients[i];
    }
  }
}

void Sparsify::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t Sparsify::dim() const { return _dim; }

std::optional<uint32_t> Sparsify::nonzeros(const ComputationList& inputs,
                                           bool use_sparsity) const {
  (void)inputs;
  if (use_sparsity) {
    return _dim * _sparsity;
  }
  return _dim;
}

void Sparsify::initOptimizer() {}

void Sparsify::summary(std::ostream& summary, const ComputationList& inputs,
                       const Computation* output) const {
  summary << "Sparsify(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << "[sparsity=" << _sparsity << "]";
}

ComputationPtr Sparsify::apply(ComputationPtr input) {
  if (dim() == 0) {
    _dim = input->dim();
  }
  if (input->dim() != _dim) {
    throw std::invalid_argument(
        "Cannot apply Sparsify expecting input with dim " +
        std::to_string(_dim) + " to input with dim " +
        std::to_string(input->dim()) + ".");
  }

  return Computation::make(shared_from_this(), {std::move(input)});
}

}  // namespace thirdai::bolt