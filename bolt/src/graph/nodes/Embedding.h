#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include "TokenInput.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt {

class EmbeddingNode final : public Node,
                            public std::enable_shared_from_this<EmbeddingNode> {
 public:
  EmbeddingNode(uint32_t num_embedding_lookups, uint32_t lookup_size,
                uint32_t log_embedding_block_size)
      : _embedding_layer(nullptr),
        _config(EmbeddingLayerConfig(
            /* num_embedding_lookups= */ num_embedding_lookups,
            /* lookup_size= */ lookup_size,
            /* log_embedding_block_size= */ log_embedding_block_size)),
        _outputs(std::nullopt),
        _token_input(nullptr) {}

  uint32_t outputDim() const final {
    NodeState node_state = getState();
    if (node_state == NodeState::Constructed ||
        node_state == NodeState::PredecessorsSet) {
      return _config->num_embedding_lookups * _config->lookup_size;
    }
    return _embedding_layer->getEmbeddingDim();
  }

  std::shared_ptr<EmbeddingNode> addInput(TokenInputPtr input) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "EmbeddingNodes have exactly one TokenInput node as input and the "
          "addInput function cannot be called more than once.");
    }

    _token_input = std::move(input);

    return shared_from_this();
  }

  bool isInputNode() const final { return false; }

  void initTrainDatastructures() final {
    _embedding_layer->initTrainDatastructures();
  }

  std::string type() const final { return "embedding"; }

  NodeState getState() const final {
    if (_token_input == nullptr && _embedding_layer == nullptr &&
        !_outputs.has_value()) {
      return NodeState::Constructed;
    }
    if (_token_input != nullptr && _embedding_layer == nullptr &&
        !_outputs.has_value()) {
      return NodeState::PredecessorsSet;
    }
    if (_token_input != nullptr && _embedding_layer != nullptr &&
        !_outputs.has_value()) {
      return NodeState::Compiled;
    }
    if (_token_input != nullptr && _embedding_layer != nullptr &&
        _outputs.has_value()) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "EmbeddingNode is in an invalid internal state");
  }

 private:
  void compileImpl() final {
    assert(_config.has_value());
    _embedding_layer = std::make_shared<EmbeddingLayer>(_config.value());
    _config = std::nullopt;
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)use_sparsity;

    _embedding_layer->initializeLayer(batch_size);
    _outputs = _embedding_layer->createBatchState(batch_size);
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;

    _embedding_layer->forward(
        /* vec_index= */ vec_index,
        /* tokens= */ _token_input->getTokens(vec_index),
        /* output= */ (*_outputs)[vec_index]);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    _embedding_layer->backpropagate(
        /* vec_index= */ vec_index,
        /* output= */ (*_outputs)[vec_index]);
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    _embedding_layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2,
                                       EPS);
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    assert(getState() == NodeState::PreparedForBatchProcessing);

    return (*_outputs)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

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

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    (void)detailed;
    summary << _token_input->name() << " -> " << name() << ": (Embedding):";
    _embedding_layer->buildLayerSummary(summary);
  }

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

  TokenInputPtr _token_input;
};

using EmbeddingNodePtr = std::shared_ptr<EmbeddingNode>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::EmbeddingNode)
