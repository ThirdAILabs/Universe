#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class Transpose final : public Op,
                        public std::enable_shared_from_this<Transpose> {
 public:
  static std::shared_ptr<Transpose> make(uint32_t rows, uint32_t columns);

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

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  autograd::ComputationPtr apply(autograd::ComputationPtr input);

 private:
  Transpose() {}

  Transpose(uint32_t rows, uint32_t columns);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  uint32_t _rows, _columns;
};

using TransposePtr = std::shared_ptr<Transpose>;

}  // namespace thirdai::bolt::nn::ops