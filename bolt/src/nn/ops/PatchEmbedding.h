#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt {

class PatchEmbedding final
    : public Op,
      public std::enable_shared_from_this<PatchEmbedding> {
 private:
  PatchEmbedding(size_t emb_dim, size_t patch_dim, size_t n_patches,
                 float sparsity, const std::string& activation,
                 SamplingConfigPtr sampling = nullptr, bool use_bias = true,
                 size_t rebuild_hash_tables = 4,
                 size_t reconstruct_hash_functions = 100);

 public:
  static auto make(size_t emb_dim, size_t patch_dim, size_t n_patches,
                   float sparsity, const std::string& activation,
                   SamplingConfigPtr sampling = nullptr, bool use_bias = true,
                   size_t rebuild_hash_tables = 4,
                   size_t reconstruct_hash_functions = 100) {
    return std::shared_ptr<PatchEmbedding>(
        new PatchEmbedding(emb_dim, patch_dim, n_patches, sparsity, activation,
                           std::move(sampling), use_bias, rebuild_hash_tables,
                           reconstruct_hash_functions));
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

  void switchToSgd() final { _kernel->switchToSgd(); }

  ComputationPtr apply(ComputationPtr input);

  void setWeights(const float* new_weights);

  void setBiases(const float* new_biases);

  void setHashTable(hashing::HashFunctionPtr hash_fn,
                    hashtable::SampledHashTablePtr hash_table);

  uint32_t patchEmbeddingDim() const;

  uint32_t patchDim() const;

 private:
  size_t patchNonzeros(bool use_sparsity) const;

  std::unique_ptr<FullyConnectedLayer> _kernel;
  size_t _n_patches;

  size_t _rebuild_hash_tables;
  size_t _reconstruct_hash_functions;
  size_t _updates_since_rebuild_hash_tables;
  size_t _updates_since_reconstruct_hash_functions;

  PatchEmbedding() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Op>(this), _kernel, _n_patches,
            _rebuild_hash_tables, _reconstruct_hash_functions,
            _updates_since_rebuild_hash_tables,
            _updates_since_reconstruct_hash_functions);
  }
};

using PatchEmbeddingPtr = std::shared_ptr<PatchEmbedding>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::PatchEmbedding)