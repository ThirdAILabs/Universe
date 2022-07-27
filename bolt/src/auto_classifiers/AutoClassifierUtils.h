#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/Datasets.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>

namespace thirdai::bolt {

class AutoClassifierUtils {
 public:
  static std::shared_ptr<FullyConnectedNetwork> createNetwork(
      uint64_t input_dim, uint32_t n_classes, const std::string& model_size);

  static std::shared_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  loadStreamingDataset(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t batch_size = 256);

  static void train(
      std::shared_ptr<FullyConnectedNetwork>& model,
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t epochs, float learning_rate);

  static void predict(
      std::shared_ptr<FullyConnectedNetwork>& model,
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      const std::optional<std::string>& output_filename,
      const std::vector<std::string>& class_id_to_class_name);

  static uint32_t getHiddenLayerSize(const std::string& model_size,
                                     uint64_t n_classes, uint64_t input_dim);

  static float getHiddenLayerSparsity(uint64_t layer_size);

  static uint64_t getMemoryBudget(const std::string& model_size);

  static std::optional<uint64_t> getSystemRam();

  static bool canLoadDatasetInMemory(const std::string& filename);
};

}  // namespace thirdai::bolt