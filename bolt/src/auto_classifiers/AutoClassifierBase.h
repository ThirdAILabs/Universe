#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>

namespace thirdai::bolt {

class AutoClassifierBase {
 public:
  AutoClassifierBase(uint64_t input_dim, uint32_t n_classes,
                     const std::string& model_size);

  AutoClassifierBase(
      uint64_t input_dim,
      std::vector<std::pair<uint32_t, float>> hidden_layer_configs,
      uint32_t output_layer_size, float output_layer_sparsity);

  void train(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t epochs, float learning_rate,
      bool prepare_for_sparse_inference = true,
      const std::vector<std::string>& metrics = {"categorical_accuracy"});

  InferenceResult predict(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      const std::optional<std::string>& output_filename,
      const std::vector<std::string>& class_id_to_class_name,
      bool use_sparse_inference = true,
      const std::vector<std::string>& metrics = {"categorical_accuracy"});

  BoltVector predictSingle(std::vector<BoltVector>&& test_data,
                           std::vector<std::vector<uint32_t>>&& test_tokens,
                           bool use_sparse_inference);

  static uint32_t getHiddenLayerSize(const std::string& model_size,
                                     uint64_t n_classes, uint64_t input_dim);

  static bool canLoadDatasetInMemory(const std::string& filename);

  static BoltGraphPtr buildModel(
      uint32_t input_dim,
      std::vector<std::pair<uint32_t, float>>& hidden_layer_configs,
      uint32_t output_layer_size, float output_layer_sparsity);

 private:
  static std::shared_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  loadStreamingDataset(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t batch_size = 256);

  static float getHiddenLayerSparsity(uint64_t layer_size);

  static uint64_t getMemoryBudget(const std::string& model_size);

  static std::optional<uint64_t> getSystemRam();

  // Private constructor for cereal
  AutoClassifierBase() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model);
  }

  BoltGraphPtr _model;
};

}  // namespace thirdai::bolt