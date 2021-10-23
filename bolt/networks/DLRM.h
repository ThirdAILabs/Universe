#pragma once

#include "../../utils/dataset/batch_types/ClickThroughBatch.h"
#include "../layers/EmbeddingLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include <vector>

namespace thirdai::bolt {

class DLRM {
 public:
  DLRM(EmbeddingLayerConfig embedding_config,
       FullyConnectedLayerConfig dense_feature_layer_config,
       std::vector<FullyConnectedLayerConfig> fc_layer_configs,
       uint32_t input_dim);

  void train(uint32_t batch_size, const std::string& train_data,
             const std::string& test_data, float learning_rate, uint32_t epochs,
             uint32_t dense_features, uint32_t categorical_features,
             uint32_t rehash = 0, uint32_t rebuild = 0,
             uint32_t max_test_batches = std::numeric_limits<uint32_t>::max());

 private:
  void processTrainingBatch(const utils::ClickThroughBatch& batch, float lr);

  uint32_t processTestBatch(const utils::ClickThroughBatch& batch);

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