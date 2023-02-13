#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <limits>
#include <memory>

namespace thirdai::bolt::nn::ops {

class Input final : public Op, public std::enable_shared_from_this<Input> {
 public:
  // TODO(Nicholas) add nonzeros as option.
  static autograd::ComputationPtr make(uint32_t dim);

  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

 private:
  Input(uint32_t dim, std::optional<uint32_t> nonzeros);

  uint32_t _dim;
  std::optional<uint32_t> _nonzeros;
};

using InputPtr = std::shared_ptr<Input>;

}  // namespace thirdai::bolt::nn::ops