#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class Concatenate final : public Op,
                          public std::enable_shared_from_this<Concatenate> {
 public:
  static std::shared_ptr<Concatenate> make();

  tensor::ActivationTensorPtr apply(const tensor::TensorList& inputs);

  void forward(const tensor::TensorList& inputs,
               tensor::ActivationTensor* output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(tensor::TensorList& inputs,
                     tensor::ActivationTensor* output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final {
    (void)learning_rate;
    (void)train_steps;
  }

  uint32_t numNonzerosInOutput(const tensor::TensorList& inputs,
                               bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  void summary(std::ostream& summary, const tensor::TensorList& inputs,
               const tensor::ActivationTensor* output) const final;

 private:
  Concatenate();

  std::vector<uint32_t> _input_dims;
  std::vector<uint32_t> _neuron_offsets;
};

}  // namespace thirdai::bolt::nn::ops