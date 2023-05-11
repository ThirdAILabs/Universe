#include "Sum.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextSumOpName() {
  static uint32_t constructed = 0;
  return "sum_" + std::to_string(++constructed);
}

Sum::Sum() : Op(nextSumOpName()) {}

std::shared_ptr<Sum> Sum::make() { return std::shared_ptr<Sum>(new Sum()); }

void Sum::forward(const autograd::ComputationList& inputs,
                  tensor::TensorPtr& output, uint32_t index_in_batch,
                  bool training) {
  assert(inputs.size() == 1);
  (void)training;

  const auto& input = inputs.at(0)->tensor();

  uint32_t seq_len = input->dims3d().at(1);

  BoltVector& output_vec = output->index2d(index_in_batch);

  std::fill_n(output_vec.activations, output_vec.len, 0);

  for (uint32_t i = 0; i < seq_len; i++) {
    const BoltVector& input_vec = input->index3d(index_in_batch, i);

    if (input_vec.isDense()) {
      for (uint32_t j = 0; j < input_vec.len; j++) {
        output_vec.activations[j] += input_vec.activations[j];
      }
    } else {
      for (uint32_t j = 0; j < input_vec.len; j++) {
        output_vec.activations[input_vec.active_neurons[j]] +=
            input_vec.activations[j];
      }
    }
  }
}

void Sum::backpropagate(autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  const auto& input = inputs.at(0)->tensor();

  uint32_t seq_len = input->dims3d().at(1);

  const BoltVector& output_vec = output->index2d(index_in_batch);

  for (uint32_t i = 0; i < seq_len; i++) {
    BoltVector& input_vec = input->index3d(index_in_batch, i);

    if (input_vec.isDense()) {
      for (uint32_t j = 0; j < input_vec.len; j++) {
        input_vec.gradients[j] += output_vec.gradients[j];
      }
    } else {
      for (uint32_t j = 0; j < input_vec.len; j++) {
        input_vec.gradients[j] +=
            output_vec.gradients[input_vec.active_neurons[j]];
      }
    }
  }
}

tensor::Dims Sum::dims(const autograd::ComputationList& inputs) const {
  assert(inputs.size() == 1);

  return {inputs.at(0)->dims().back()};
}

std::optional<uint32_t> Sum::nonzeros(const autograd::ComputationList& inputs,
                                      bool use_sparsity) const {
  (void)use_sparsity;
  return inputs.at(0)->dims().back();
}

void Sum::summary(std::ostream& summary,
                  const autograd::ComputationList& inputs,
                  const autograd::Computation* output) const {
  summary << "Sum(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
}

autograd::ComputationPtr Sum::apply(autograd::ComputationPtr input) {
  if (input->dims().size() != 3) {
    throw std::invalid_argument("Sum op can only be applied to 3d input.");
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Sum::serialize(cereal::BinaryInputArchive&);
template void Sum::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Sum::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Sum)