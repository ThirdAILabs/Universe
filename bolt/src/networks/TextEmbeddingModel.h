#pragma once

#include "FullyConnectedNetwork.h"
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/networks/Model.h>
#include <dataset/src/batch_types/MaskedSentenceBatch.h>
#include <vector>

namespace thirdai::bolt {

class TextEmbeddingModel final : public Model<dataset::MaskedSentenceBatch> {
 public:
  void forward(uint32_t batch_index, const dataset::MaskedSentenceBatch& input,
               BoltVector& output, const BoltVector* labels) final;

  // Backpropagates gradients through the network
  void backpropagate(uint32_t batch_index, dataset::MaskedSentenceBatch& input,
                     BoltVector& output) final;

  // Performs parameter updates for the network.
  void updateParameters(float learning_rate, uint32_t iter) final;

  // Called for network to allocate any necessary state to store activations and
  // gradients.
  void initializeNetworkState(uint32_t batch_size, bool force_dense) final;

  // Construct new hash functions (primarly for fully connected layers).
  void reBuildHashFunctions() final {
    _bottom_embedding_layers.reBuildHashFunctions();
    for (auto& layer : _token_embedding_layers) {
      layer.reBuildHashFunction();
    }
    _classifier.reBuildHashFunctions();
  }

  // Rebuild any hash tables (primarly for fully connected layers).
  void buildHashTables() final {
    _bottom_embedding_layers.buildHashTables();
    for (auto& layer : _token_embedding_layers) {
      layer.buildHashTables();
    }
    _classifier.buildHashTables();
  }

  // Shuffles neurons for random sampling.
  void shuffleRandomNeurons() final {
    _bottom_embedding_layers.shuffleRandomNeurons();
    for (auto& layer : _token_embedding_layers) {
      layer.shuffleRandNeurons();
    }
    _classifier.shuffleRandomNeurons();
  }

  // Allocates storage for activations and gradients for output layer.
  BoltBatch getOutputs(uint32_t batch_size, bool force_dense) final {
    return _classifier.getOutputs(batch_size, force_dense);
  }

  // Gets the dimension of the output layer.
  uint32_t getOutputDim() const final { return _classifier.getOutputDim(); }

  uint32_t getInferenceOutputDim() const final {
    return _classifier.getInferenceOutputDim();
  }

 private:
  FullyConnectedNetwork _bottom_embedding_layers;
  std::vector<FullyConnectedLayer> _token_embedding_layers;
  FullyConnectedNetwork _classifier;

  BoltBatch _bottom_embedding_output;
  BoltBatch _token_embedding_output;

  std::vector<bool> _token_embedding_layers_used;
};

}  // namespace thirdai::bolt