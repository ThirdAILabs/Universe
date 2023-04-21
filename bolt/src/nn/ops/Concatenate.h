#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <utils/CerealWrapper.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class Concatenate final : public Op,
                          public std::enable_shared_from_this<Concatenate> {
 public:
  static std::shared_ptr<Concatenate> make();

  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final {
    (void)learning_rate;
    (void)train_steps;
  }

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; };

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  autograd::ComputationPtr apply(const autograd::ComputationList& inputs);

 private:
  Concatenate();

  std::vector<uint32_t> _input_dims;
  std::vector<uint32_t> _neuron_offsets;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using ConcatenatePtr = std::shared_ptr<Concatenate>;

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE_HEADER(thirdai::bolt::nn::ops::Concatenate)