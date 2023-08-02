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
  (void)training;

  const BoltVector& input_vec =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& output_vec = output->getVector(index_in_batch);

  if (!input_vec.isDense()) {
    std::copy(input_vec.active_neurons,
              input_vec.active_neurons + input_vec.len,
              output_vec.active_neurons);
  }

  for (uint32_t i = 0; i < input_vec.len; i++) {
    output_vec.activations[i] = std::tanh(input_vec.activations[i]);
  }
}

void Tanh::backpropagate(autograd::ComputationList& inputs,
                         tensor::TensorPtr& output, uint32_t index_in_batch) {
  BoltVector& input_vec = inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& output_vec = output->getVector(index_in_batch);

  for (uint32_t i = 0; i < input_vec.len; i++) {
    float tanh = output_vec.activations[i];
    input_vec.gradients[i] += (1 - tanh * tanh) * output_vec.gradients[i];
  }
}

void Tanh::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t Tanh::dim() const { return _dim; }

std::optional<uint32_t> Tanh::nonzeros(const autograd::ComputationList& inputs,
                                       bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void Tanh::disableSparseParameterUpdates() {}

void Tanh::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> Tanh::gradients() { return {}; }

std::vector<std::vector<float>*> Tanh::parameters() { return {}; }

bolt_proto::Op Tanh::toProto(bool with_optimizer) const {
  (void)with_optimizer;

  bolt_proto::Op op;

  op.set_name(name());
  op.mutable_tanh();

  return op;
}

void Tanh::summary(std::ostream& summary,
                   const autograd::ComputationList& inputs,
                   const autograd::Computation* output) const {
  summary << "Tanh(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
}

autograd::ComputationPtr Tanh::apply(autograd::ComputationPtr input) {
  if (dim() == 0) {
    _dim = input->dim();
  } else {
    if (dim() != input->dim()) {
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