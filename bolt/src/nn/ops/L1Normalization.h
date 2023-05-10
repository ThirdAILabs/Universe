#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class L1Normalization final
    : public Op,
      public std::enable_shared_from_this<L1Normalization> {
 public:
  static std::shared_ptr<L1Normalization> make();

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

  autograd::ComputationPtr apply(autograd::ComputationPtr input);

 private:
  L1Normalization();

  static void l1Normalization(const BoltVector& input, BoltVector& output);

  static void l1NormalizationGradient(BoltVector& input,
                                      const BoltVector& output);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using L1NormalizationPtr = std::shared_ptr<L1Normalization>;

}  // namespace thirdai::bolt::nn::ops