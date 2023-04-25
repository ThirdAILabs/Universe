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

  static autograd::ComputationPtr make(tensor::Dims dims);

  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  tensor::Dims dims(const autograd::ComputationList& inputs) const final;

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final { return {}; };

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

 private:
  Input(tensor::Dims dims, std::optional<uint32_t> nonzeros);

  tensor::Dims _dims;
  std::optional<uint32_t> _nonzeros;

  Input() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using InputPtr = std::shared_ptr<Input>;

}  // namespace thirdai::bolt::nn::ops