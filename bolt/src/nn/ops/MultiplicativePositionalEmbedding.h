#pragma once
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <memory>

namespace thirdai::bolt::nn::ops {
class MultiplicativePosEmbedding final
    : public Op,
      public std::enable_shared_from_this<MultiplicativePosEmbedding> {
 public:
  static std::shared_ptr<MultiplicativePosEmbedding> make(
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

  std::shared_ptr<MultiplicativePosEmbedding> duplicateWithNewReduction(
      const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input);

 private:
  MultiplicativePosEmbedding(uint64_t num_embedding_lookups,
                             uint64_t lookup_size,
                             uint64_t log_embedding_block_size,
                             const std::string& reduction,
                             std::optional<uint64_t> num_tokens_per_input,
                             uint64_t update_chunk_size);

  MultiplicativePosEmbedding(std::unique_ptr<EmbeddingLayer>&& pos_kernel,
                             std::unique_ptr<EmbeddingLayer>&& token_kernel,
                             const std::string& name)
      : Op(name),
        _pos_kernel(std::move(pos_kernel)),
        _token_kernel(std::move(token_kernel)) {}

  std::unique_ptr<EmbeddingLayer> _pos_kernel, _token_kernel;

  MultiplicativePosEmbedding() {}

  friend class cereal::access;

  // We use save/load instead of serialize so we can ensure the optimizer is
  // initialized when the model is loaded.
  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};
using MultiplicativePosEmbeddingPtr =
    std::shared_ptr<MultiplicativePosEmbedding>;
}  // namespace thirdai::bolt::nn::ops