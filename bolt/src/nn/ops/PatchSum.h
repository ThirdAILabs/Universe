#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt {

class PatchSum final : public Op,
                       public std::enable_shared_from_this<PatchSum> {
 private:
  PatchSum(size_t n_patches, size_t patch_dim);

  explicit PatchSum(const ar::Archive& archive);

 public:
  static auto make(size_t n_patches, size_t patch_dim) {
    return std::shared_ptr<PatchSum>(new PatchSum(n_patches, patch_dim));
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

  static std::shared_ptr<PatchSum> fromArchive(const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  ComputationPtr apply(ComputationPtr input);

  static std::string type() { return "patch_sum"; }

 private:
  size_t _n_patches, _patch_dim;

  PatchSum() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Op>(this), _n_patches, _patch_dim);
  }
};

using PatchSumPtr = std::shared_ptr<PatchSum>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::PatchSum)