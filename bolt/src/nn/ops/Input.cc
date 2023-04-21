#include "Input.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::nn::ops {

std::string nextInputName() {
  static uint32_t constructed = 0;
  return "input_" + std::to_string(++constructed);
}

Input::Input(uint32_t dim, std::optional<uint32_t> nonzeros)
    : Op(nextInputName()), _dim(dim), _nonzeros(nonzeros) {}

autograd::ComputationPtr Input::make(uint32_t dim) {
  return autograd::Computation::make(
      std::shared_ptr<Input>(new Input(dim, /* nonzeros= */ std::nullopt)), {});
}

void Input::forward(const autograd::ComputationList& inputs,
                    tensor::TensorPtr& output, uint32_t index_in_batch,
                    bool training) {
  (void)inputs;
  (void)output;
  (void)index_in_batch;
  (void)training;

  throw std::runtime_error("Forward should not be called on input op.");
}

void Input::backpropagate(autograd::ComputationList& inputs,
                          tensor::TensorPtr& output, uint32_t index_in_batch) {
  (void)inputs;
  (void)output;
  (void)index_in_batch;

  throw std::runtime_error("Backpropagate should not be called on input op.");
}

void Input::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
  throw std::runtime_error(
      "UpdateParameters should not be called on input op.");
}

uint32_t Input::dim() const { return _dim; }

std::optional<uint32_t> Input::nonzeros(const autograd::ComputationList& inputs,
                                        bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;
  return _nonzeros;
}

void Input::disableSparseParameterUpdates() {}

void Input::summary(std::ostream& summary,
                    const autograd::ComputationList& inputs,
                    const autograd::Computation* output) const {
  (void)inputs;
  summary << "Input(" << name() << ") -> " << output->name();
}

template void Input::serialize(cereal::BinaryInputArchive&);
template void Input::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Input::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _dim, _nonzeros);
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE_SOURCE(thirdai::bolt::nn::ops::Input)