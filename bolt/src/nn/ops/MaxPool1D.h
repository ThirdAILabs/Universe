#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>
#include <optional>

namespace thirdai::bolt {

class MaxPool1D final : public Op,
                        public std::enable_shared_from_this<MaxPool1D> {
 private:
  explicit MaxPool1D(size_t window_size);

 public:
  static auto make(size_t window_size) {
    return std::shared_ptr<MaxPool1D>(new MaxPool1D(window_size));
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  ComputationPtr apply(ComputationPtr input);

 private:
  size_t _output_dim = 0;
  size_t _window_size;

  MaxPool1D() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Op>(this), _output_dim, _window_size);
  }
};

using MaxPool1DPtr = std::shared_ptr<MaxPool1D>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::MaxPool1D)