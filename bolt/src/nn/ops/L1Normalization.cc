#include "L1Normalization.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::ops {

std::string nextL1NormalizationName() {
  static uint32_t constructed = 0;
  return "l1_norm_" + std::to_string(++constructed);
}

L1Normalization::L1Normalization() : Op(nextL1NormalizationName()) {}

std::shared_ptr<L1Normalization> L1Normalization::make() {
  return std::shared_ptr<L1Normalization>(new L1Normalization());
}

void L1Normalization::forward(const autograd::ComputationList& inputs,
                              tensor::TensorPtr& output,
                              uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const tensor::TensorPtr& input = inputs.at(0)->tensor();

  uint32_t len = output->dims3d().at(1);

  for (uint32_t i = 0; i < len; i++) {
    l1Normalization(input->at_3d(index_in_batch, i),
                    output->at_3d(index_in_batch, i));
  }
}

constexpr float relu(float x) { return std::max(x, 0.F); }

void L1Normalization::l1Normalization(const BoltVector& input,
                                      BoltVector& output) {
  assert(input.len == output.len);

  /**
   * We apply a relu here to avoid having to use an absolute value when
   * computing the l1 norm. This is because if we include the absolute value
   * then it requires O(N^2) computations to compute the jacobian in the
   * backward pass. Without the absolute value we can make use of repeating
   * structure in the jabobian to compute it in O(N) time.
   */
  float l1_norm = 0.0;
  for (uint32_t i = 0; i < input.len; i++) {
    l1_norm += relu(input.activations[i]);
  }

  for (uint32_t i = 0; i < input.len; i++) {
    output.activations[i] = relu(input.activations[i]) / l1_norm;
  }

  if (!input.isDense()) {
    std::copy(input.active_neurons, input.active_neurons + input.len,
              output.active_neurons);
  }
}

void L1Normalization::backpropagate(autograd::ComputationList& inputs,
                                    tensor::TensorPtr& output,
                                    uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  const tensor::TensorPtr& input = inputs.at(0)->tensor();

  uint32_t len = output->dims3d().at(1);

  for (uint32_t i = 0; i < len; i++) {
    l1NormalizationGradient(input->at_3d(index_in_batch, i),
                            output->at_3d(index_in_batch, i));
  }
}

constexpr float reluGrad(float x) { return x > 0 ? 1 : 0; }

void L1Normalization::l1NormalizationGradient(BoltVector& input,
                                              const BoltVector& output) {
  assert(input.len == output.len);

  float l1_norm = 0.0;
  float sum_grad_times_output = 0.0;

  for (uint32_t i = 0; i < output.len; i++) {
    sum_grad_times_output += output.gradients[i] * output.activations[i];
    l1_norm += relu(input.activations[i]);
  }

  for (uint32_t i = 0; i < input.len; i++) {
    float grad = (output.gradients[i] - sum_grad_times_output) / l1_norm;
    input.gradients[i] = grad * reluGrad(input.activations[i]);
  }
}

tensor::Dims L1Normalization::dims(
    const autograd::ComputationList& inputs) const {
  assert(inputs.size() == 1);
  return inputs.at(0)->dims();
}

std::optional<uint32_t> L1Normalization::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  assert(inputs.size() == 1);
  return inputs.at(0)->nonzeros(use_sparsity);
}

void L1Normalization::summary(std::ostream& summary,
                              const autograd::ComputationList& inputs,
                              const autograd::Computation* output) const {
  summary << "L1Normalization(" << name() << "): " << inputs.at(0)->name()
          << " -> " << output->name() << std::endl;
}

autograd::ComputationPtr L1Normalization::apply(
    autograd::ComputationPtr input) {
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void L1Normalization::serialize(cereal::BinaryInputArchive&);
template void L1Normalization::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void L1Normalization::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this));
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::L1Normalization)