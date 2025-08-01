#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <archive/src/Archive.h>
#include <memory>

namespace thirdai::bolt {

/**
 * This op applies a quantile based thresholding to each window in the input.
 * For each window the op zeros out everything except the frac largest elements.
 * This yields an output vector of equivalent dimension, just with the elements
 * zeroed out.
 */
class QuantileMixing final
    : public Op,
      public std::enable_shared_from_this<QuantileMixing> {
 private:
  explicit QuantileMixing(size_t window_size, float frac);

  explicit QuantileMixing(const ar::Archive& archive);

 public:
  static auto make(size_t window_size, float frac) {
    return std::shared_ptr<QuantileMixing>(
        new QuantileMixing(window_size, frac));
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory,
                     bool replace_existing_optimizer) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::shared_ptr<QuantileMixing> fromArchive(
      const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  ComputationPtr apply(ComputationPtr input);

  static std::string type() { return "quantile_mixing"; }

 private:
  size_t _output_dim = 0;
  size_t _window_size;
  float _frac;

  QuantileMixing() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Op>(this), _frac, _output_dim, _window_size);
  }
};

using QuantileMixingPtr = std::shared_ptr<QuantileMixing>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::QuantileMixing)