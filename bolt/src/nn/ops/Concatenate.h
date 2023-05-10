#pragma once

#include <bolt/src/nn/ops/Op.h>
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

  tensor::Dims dims(const autograd::ComputationList& inputs) const final;

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; };

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  autograd::ComputationPtr apply(const autograd::ComputationList& inputs);

 private:
  Concatenate();

  void forwardHelper(const autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch,
                     uint32_t index) const;

  static void backpropagateHelper(autograd::ComputationList& inputs,
                                  const tensor::TensorPtr& output,
                                  uint32_t index_in_batch, uint32_t index);

  std::vector<uint32_t> _input_dims;
  std::vector<uint32_t> _neuron_offsets;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using ConcatenatePtr = std::shared_ptr<Concatenate>;

}  // namespace thirdai::bolt::nn::ops