#pragma once

#include "../../utils/dataset/batch_types/ClickThroughBatch.h"
#include "../layers/FullyConnectedLayer.h"
#include <vector>

namespace thirdai::bolt {

class TwoTower {
 public:
  TwoTower(std::vector<FullyConnectedLayerConfig> query_layer_configs,
           std::vector<FullyConnectedLayerConfig> candid_layer_configs,
           uint32_t input_dim);

  void train(u_int32_t batch_size, const std::string& train_data,
             const std::string& test_data, float learning_rate, uint32_t epochs,
             uint32_t rehash = 0, uint32_t rebuild = 0,
             uint32_t max_test_batches = std::numeric_limits<uint32_t>::max());

 private:
  void processTrainingBatch(const Batch& batch, const Batch& batch);

  uint32_t processTestBatch(const ClickThroughBatch& batch);

  void initializeNetworksForBatchSize(uint32_t batch_size);

  void reBuildHashFunctions();

  void buildHashTables();

  uint32_t _num_layers;
  FullyConnectedLayer** _query_layers;
  FullyConnectedLayer** _candid_layers;
  std::vector<FullyConnectedLayerConfig> _query_layer_configs;
  std::vector<FullyConnectedLayerConfig> _candid_layer_configs;

  uint32_t _iter;

  std::vector<float> _accuracy_per_epoch;
  std::vector<int64_t> _time_per_epoch;
  float _final_accuracy;
};

}  // namespace thirdai::bolt