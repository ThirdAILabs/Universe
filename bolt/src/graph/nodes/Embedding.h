#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include "Input.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt_vector/src/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <memory>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt {

class EmbeddingNode final : public Node,
                            public std::enable_shared_from_this<EmbeddingNode> {
 private:
  EmbeddingNode(uint64_t num_embedding_lookups, uint64_t lookup_size,
                uint64_t log_embedding_block_size, uint64_t chunk_size,
                const std::string& reduction,
                std::optional<uint64_t> num_tokens_per_input);

 public:
  static std::shared_ptr<EmbeddingNode> make(
      uint64_t num_embedding_lookups, uint32_t lookup_size,
      uint64_t log_embedding_block_size, uint64_t chunk_size,
      const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input = std::nullopt) {
    return std::shared_ptr<EmbeddingNode>(new EmbeddingNode(
        num_embedding_lookups, lookup_size, log_embedding_block_size,
        chunk_size, reduction, num_tokens_per_input));
  }

  uint32_t outputDim() const final;

  std::shared_ptr<EmbeddingNode> addInput(InputPtr input);

  bool isInputNode() const final { return false; }

  void initOptimizer() final { _embedding_layer->initOptimizer(); }

  std::string type() const final { return "embedding"; }

  NodeState getState() const final;

  void disableSparseParameterUpdates() final;

  std::vector<float>& getRawEmbeddingBlock() {
    return _embedding_layer->getRawEmbeddingBlock();
  }

  std::vector<float>& getRawEmbeddingBlockGradient() {
    return _embedding_layer->getRawEmbeddingBlockGradient();
  }

  bool hasParameters() final { return true; }

 private:
  void compileImpl() final;

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final;

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  uint32_t numNonzerosInOutputImpl() const final {
    // The embedding is dense so we can just return the result of outputDim.
    return outputDim();
  }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_token_input};
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  // Private constructor for cereal.
  EmbeddingNode() : _config(std::nullopt), _outputs(std::nullopt) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _embedding_layer, _config,
            _token_input);
  }

  // This field will be a nullptr except for when the node is in the
  // PrepareForBatchProcessing state.
  std::shared_ptr<EmbeddingLayer> _embedding_layer;

  std::optional<EmbeddingLayerConfig> _config;

  // This field will be std::nullopt except for when the node is in the
  // PrepareForBatchProcessing state.
  std::optional<BoltBatch> _outputs;

  InputPtr _token_input;
};

using EmbeddingNodePtr = std::shared_ptr<EmbeddingNode>;

}  // namespace thirdai::bolt
