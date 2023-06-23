#pragma once

#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class RobeZ final : public Op, public std::enable_shared_from_this<RobeZ> {
 public:
  static std::shared_ptr<RobeZ> make(
      uint64_t num_embedding_lookups, uint64_t lookup_size,
      uint64_t log_embedding_block_size, const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input = std::nullopt,
      uint64_t update_chunk_size = DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE);

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

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  autograd::ComputationPtr apply(autograd::ComputationPtr input);

  std::shared_ptr<RobeZ> duplicateWithNewReduction(
      const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input);

 private:
  RobeZ(uint64_t num_embedding_lookups, uint64_t lookup_size,
        uint64_t log_embedding_block_size, const std::string& reduction,
        std::optional<uint64_t> num_tokens_per_input,
        uint64_t update_chunk_size);

  RobeZ(std::unique_ptr<EmbeddingLayer>&& kernel, const std::string& name)
      : Op(name), _kernel(std::move(kernel)) {}

  std::unique_ptr<EmbeddingLayer> _kernel;

  RobeZ() {}

  friend class cereal::access;

  // We use save/load instead of serialize so we can ensure the optimizer is
  // initialized when the model is loaded.
  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};

using RobeZPtr = std::shared_ptr<RobeZ>;

}  // namespace thirdai::bolt::nn::ops