#pragma once

#include "TokenInput.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt {

class EmbeddingNode final : public Node {
 public:
  EmbeddingNode(uint32_t num_embedding_lookups, uint32_t lookup_size,
                uint32_t log_embedding_block_size)
      : _embedding_layer(nullptr),
        _config(/* num_embedding_lookups= */ num_embedding_lookups,
                /* lookup_size= */ lookup_size,
                /* log_embedding_block_size= */ log_embedding_block_size),
        _outputs(std::nullopt),
        _token_input(nullptr) {}

  uint32_t outputDim() const final {
    return _embedding_layer->getEmbeddingDim();
  }

  uint32_t numNonzerosInOutput() const final {
    // The embedding is dense so we can just return the result of outputDim.
    return outputDim();
  }

  std::vector<NodePtr> getPredecessors() const final { return {}; }

  void addInput(TokenInputPtr input) {
    if (predecessorsSet()) {
      throw exceptions::NodeStateMachineError(
          "Embedding expected to have exactly one input, and "
          "addInput cannot be called twice.");
    }

    _token_input = std::move(input);
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    return {};
  }

  bool isInputNode() const final { return false; }

 private:
  void initializeParametersImpl() final {
    _embedding_layer = std::make_shared<EmbeddingLayer>(_config);
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)use_sparsity;

    _embedding_layer->initializeLayer(batch_size);
    _outputs = _embedding_layer->createBatchState(batch_size);
  }

  void forwardImpl(uint32_t batch_index, const BoltVector* labels) final {
    (void)labels;

    assert(preparedForBatchProcessing());

    _embedding_layer->forward(
        /* batch_index= */ batch_index,
        /* tokens= */ _token_input->getTokens(batch_index),
        /* output= */ (*_outputs)[batch_index]);
  }

  void backpropagateImpl(uint32_t batch_index) final {
    assert(preparedForBatchProcessing());

    _embedding_layer->backpropagate(
        /* batch_index= */ batch_index,
        /* output= */ (*_outputs)[batch_index]);
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    assert(preparedForBatchProcessing());

    _embedding_layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2,
                                       EPS);
  }

  BoltVector& getOutputVectorImpl(uint32_t batch_index) final {
    assert(preparedForBatchProcessing());

    return (*_outputs)[batch_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

  bool predecessorsSet() const final { return _token_input != nullptr; }

  bool parametersInitialized() const final {
    return _embedding_layer != nullptr;
  }

  bool preparedForBatchProcessing() const final { return _outputs.has_value(); }

  std::shared_ptr<EmbeddingLayer> _embedding_layer;
  EmbeddingLayerConfig _config;
  std::optional<BoltBatch> _outputs;

  std::shared_ptr<TokenInput> _token_input;
};

}  // namespace thirdai::bolt