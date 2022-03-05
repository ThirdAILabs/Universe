#pragma once

#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
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
      dataset::InMemoryDataset<dataset::ClickThroughBatch>& train_data,
      float learning_rate, uint32_t epochs, uint32_t rehash = 0,
      uint32_t rebuild = 0);

  void predict(
       dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data,
      float* scores);

 private:
  void forward(uint32_t batch_index, const BoltVector& dense_input,
               const std::vector<uint32_t>& categorical_features,
               BoltVector& output);

  void backpropagate(uint32_t batch_index, BoltVector& dense_input,
                     BoltVector& output);

  void initializeNetworkForBatchSize(uint32_t batch_size, bool force_dense);

  void reBuildHashFunctions();

  void buildHashTables();

  EmbeddingLayer _embedding_layer;
  FullyConnectedNetwork _bottom_mlp;
  FullyConnectedNetwork _top_mlp;

  uint32_t _concat_layer_dim;
  BoltBatch _concat_layer_state;
  std::vector<BoltVector> _embedding_layer_output;
  std::vector<BoltVector> _bottom_mlp_output;

  uint32_t _iter;
  uint32_t _epoch_count;

  bool _softmax;

 protected:
  uint32_t _output_dim;
};

}  // namespace thirdai::bolt