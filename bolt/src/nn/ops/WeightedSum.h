#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class WeightedSum final : public Op,
                          public std::enable_shared_from_this<WeightedSum> {
 public:
  static std::shared_ptr<WeightedSum> make();

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

  std::vector<std::vector<float>*> gradients() final { return {}; }

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  autograd::ComputationPtr apply(const autograd::ComputationPtr& embeddings,
                                 const autograd::ComputationPtr& weights);

 private:
  WeightedSum();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using WeightedSumPtr = std::shared_ptr<WeightedSum>;

}  // namespace thirdai::bolt::nn::ops