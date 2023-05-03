#include "Tanh.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextTanhOpName() {
  static uint32_t constructed = 0;
  return "tanh_" + std::to_string(++constructed);
}

Tanh::Tanh() : Op(nextTanhOpName()) {}

std::shared_ptr<Tanh> Tanh::make() { return std::shared_ptr<Tanh>(new Tanh()); }

void Tanh::forward(const autograd::ComputationList& inputs,
                   tensor::TensorPtr& output, uint32_t index_in_batch,
                   bool training) {
  assert(inputs.size() == 1);
  (void)training;

  const auto& input = inputs.at(0)->tensor();

  uint32_t start = output->rangeStart(index_in_batch);
  uint32_t end = output->rangeEnd(index_in_batch);

  for (uint32_t i = start; i < end; i++) {
    const BoltVector& input_vec = input->getVector(i);
    BoltVector& output_vec = output->getVector(i);

    if (!input_vec.isDense()) {
      std::copy(input_vec.active_neurons,
                input_vec.active_neurons + input_vec.len,
                output_vec.active_neurons);
    }

    for (uint32_t j = 0; j < input_vec.len; j++) {
      output_vec.activations[j] = std::tanh(input_vec.activations[j]);
    }
  }
}

void Tanh::backpropagate(autograd::ComputationList& inputs,
                         tensor::TensorPtr& output, uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  const auto& input = inputs.at(0)->tensor();

  uint32_t start = output->rangeStart(index_in_batch);
  uint32_t end = output->rangeEnd(index_in_batch);

  for (uint32_t i = start; i < end; i++) {
    BoltVector& input_vec = input->getVector(i);
    const BoltVector& output_vec = output->getVector(i);

    for (uint32_t j = 0; j < input_vec.len; j++) {
      float tanh = output_vec.activations[j];
      input_vec.gradients[j] += (1 - tanh * tanh) * output_vec.gradients[j];
    }
  }
}

void Tanh::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

tensor::Dims Tanh::dims(const autograd::ComputationList& inputs) const {
  assert(inputs.size() == 1);

  return inputs.at(0)->dims();
}

std::optional<uint32_t> Tanh::nonzeros(const autograd::ComputationList& inputs,
                                       bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void Tanh::disableSparseParameterUpdates() {}

std::vector<std::vector<float>*> Tanh::gradients() { return {}; }

void Tanh::summary(std::ostream& summary,
                   const autograd::ComputationList& inputs,
                   const autograd::Computation* output) const {
  summary << "Tanh(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
}

autograd::ComputationPtr Tanh::apply(autograd::ComputationPtr input) {
  if (dim() == 0) {
    _dim = input->dims().back();
  } else {
    if (dim() != input->dims().back()) {
      throw std::invalid_argument("Dim mismatch in tanh.");
    }
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Tanh::serialize(cereal::BinaryInputArchive&);
template void Tanh::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Tanh::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim);
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Tanh)