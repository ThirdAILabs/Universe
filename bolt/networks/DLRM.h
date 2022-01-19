#pragma once

#include <bolt/layers/EmbeddingLayer.h>
#include <bolt/layers/FullyConnectedLayer.h>
#include <bolt/networks/Network.h>
#include <dataset/src/Dataset.h>
#include <vector>

namespace thirdai::bolt {

class DLRM {
 public:
  DLRM(EmbeddingLayerConfig embedding_config,
       std::vector<FullyConnectedLayerConfig> bottom_mlp_configs,
       std::vector<FullyConnectedLayerConfig> top_mlp_configs,
       uint32_t dense_feature_dim);

  void train(
      const dataset::InMemoryDataset<dataset::ClickThroughBatch>& train_data,
      float learning_rate, uint32_t epochs, uint32_t rehash = 0,
      uint32_t rebuild = 0);

  void testImpl(
      const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data,
      float* scores);

 private:
  void forward(uint32_t batch_index, const VectorState& dense_input,
               const std::vector<uint32_t>& categorical_features,
               VectorState& output);

  void backpropagate(uint32_t batch_index, VectorState& dense_input,
                     VectorState& output);

  void initializeNetworkForBatchSize(uint32_t batch_size, bool force_dense);

  void reBuildHashFunctions();

  void buildHashTables();

  EmbeddingLayer _embedding_layer;
  Network _bottom_mlp;
  Network _top_mlp;

  uint32_t _concat_layer_dim;
  BatchState _concat_layer_state;
  std::vector<VectorState> _embedding_layer_output;
  std::vector<VectorState> _bottom_mlp_output;

  uint32_t _iter;
  uint32_t _epoch_count;

  bool _softmax;

 protected:
  uint32_t _output_dim;
};

}  // namespace thirdai::bolt