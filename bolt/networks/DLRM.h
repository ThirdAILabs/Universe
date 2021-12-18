#pragma once

#include <bolt/layers/EmbeddingLayer.h>
#include <bolt/layers/FullyConnectedLayer.h>
#include <dataset/src/Dataset.h>
#include <vector>

namespace thirdai::bolt {

class DLRM {
 public:
  DLRM(EmbeddingLayerConfig embedding_config,
       FullyConnectedLayerConfig dense_feature_layer_config,
       std::vector<FullyConnectedLayerConfig> fc_layer_configs,
       uint32_t input_dim);

  void train(
      const dataset::InMemoryDataset<dataset::ClickThroughBatch>& train_data,
      float learning_rate, uint32_t epochs, uint32_t rehash = 0,
      uint32_t rebuild = 0);

  void testImpl(
      const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data,
      float* scores);

 private:
  void processTrainingBatch(const dataset::ClickThroughBatch& batch, float lr);

  void processTestBatch(const dataset::ClickThroughBatch& batch, float* scores);

  void initializeNetworkForBatchSize(uint32_t batch_size);

  void reBuildHashFunctions();

  void buildHashTables();

  EmbeddingLayer* _embedding_layer;
  FullyConnectedLayer* _dense_feature_layer;
  uint32_t _num_fc_layers;
  FullyConnectedLayer** _fc_layers;
  std::vector<FullyConnectedLayerConfig> _fc_layer_configs;

  uint32_t _concat_layer_dim;
  float** _concat_layer_activations;
  float** _concat_layer_errors;

  uint32_t _iter;

  std::vector<float> _accuracy_per_epoch;
  std::vector<int64_t> _time_per_epoch;
  float _final_accuracy;
};

}  // namespace thirdai::bolt