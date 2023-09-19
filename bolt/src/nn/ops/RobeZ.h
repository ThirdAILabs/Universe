#pragma once

#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/nn/ops/Op.h>
#include <utils/Random.h>
#include <memory>

namespace thirdai::bolt {

class RobeZ final : public Op, public std::enable_shared_from_this<RobeZ> {
 public:
  static std::shared_ptr<RobeZ> make(
      uint64_t num_embedding_lookups, uint64_t lookup_size,
      uint64_t log_embedding_block_size, const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input = std::nullopt,
      uint64_t update_chunk_size = DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE,
      uint32_t seed = global_random::nextSeed());

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

  ComputationPtr apply(const ComputationList& inputs) final;

  ComputationPtr applyUnary(ComputationPtr input);

  proto::bolt::Op* toProto(bool with_optimizer) const final;

  SerializableParameters serializableParameters(
      bool with_optimizer) const final;

  static std::shared_ptr<RobeZ> fromProto(
      const std::string& name, const proto::bolt::RobeZ& robez_proto);

  std::shared_ptr<RobeZ> duplicateWithNewReduction(
      const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input);

  const auto& kernel() const { return _kernel; }

 private:
  RobeZ(uint64_t num_embedding_lookups, uint64_t lookup_size,
        uint64_t log_embedding_block_size, const std::string& reduction,
        std::optional<uint64_t> num_tokens_per_input,
        uint64_t update_chunk_size, uint32_t seed);

  RobeZ(std::unique_ptr<EmbeddingLayer>&& kernel, const std::string& name)
      : Op(name), _kernel(std::move(kernel)) {}

  RobeZ(const std::string& name, const proto::bolt::RobeZ& robez_proto);

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

}  // namespace thirdai::bolt