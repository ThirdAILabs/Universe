#include "Sigmoid.h"
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

std::string nextSigmoidOpName() {
  static uint32_t constructed = 0;
  return "sigmoid_" + std::to_string(++constructed);
}

Sigmoid::Sigmoid() : Op(nextSigmoidOpName()) {}

std::shared_ptr<Sigmoid> Sigmoid::make() {
  return std::shared_ptr<Sigmoid>(new Sigmoid());
}

void Sigmoid::forward(const autograd::ComputationList& inputs,
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
    output_vec.activations[i] =
        1.0 / (1.0 + std::exp(-1 * input_vec.activations[i]));
    ;
  }
}

void Sigmoid::backpropagate(autograd::ComputationList& inputs,
                            tensor::TensorPtr& output,
                            uint32_t index_in_batch) {
  BoltVector& input_vec = inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& output_vec = output->getVector(index_in_batch);

  for (uint32_t i = 0; i < input_vec.len; i++) {
    float sigmoid = output_vec.activations[i];
    input_vec.gradients[i] += (1 - sigmoid) * (sigmoid)*output_vec.gradients[i];
  }
}

void Sigmoid::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t Sigmoid::dim() const { return _dim; }

std::optional<uint32_t> Sigmoid::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void Sigmoid::disableSparseParameterUpdates() {}

std::vector<std::vector<float>*> Sigmoid::gradients() { return {}; }
std::vector<std::vector<float>*> Sigmoid::parameters() { return {}; }

void Sigmoid::summary(std::ostream& summary,
                      const autograd::ComputationList& inputs,
                      const autograd::Computation* output) const {
  summary << "Sigmoid(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
}

autograd::ComputationPtr Sigmoid::apply(autograd::ComputationPtr input) {
  if (dim() == 0) {
    _dim = input->dim();
  } else {
    if (dim() != input->dim()) {
      throw std::invalid_argument("Dim mismatch in Sigmoid.");
    }
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Sigmoid::serialize(cereal::BinaryInputArchive&);
template void Sigmoid::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Sigmoid::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim);
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Sigmoid)