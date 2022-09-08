#pragma once

#include <bolt/src/graph/DistributedBoltGraph.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
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

  void train(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t epochs, float learning_rate);

  void initClassifierDistributedTraining(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t epochs, float learning_rate, int batch_size);

  DistributedTrainingContext getDistributedTrainingContext() {
    if (_distributed_train_context == nullptr) {
      throw std::runtime_error(
          "Distributed Training Context is not initialized. Call "
          "initClassifierDistributedTraining() first.");
    }
    return *(_distributed_train_context);
  }

  BoltGraph getBoltGraphModel() {
    if (_model == nullptr) {
      throw std::runtime_error("Model is not initialized.");
    }
    return *(_model);
  }

  void predict(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      const std::optional<std::string>& output_filename,
      const std::vector<std::string>& class_id_to_class_name);

  BoltVector predictSingle(std::vector<BoltVector>&& test_data,
                           bool use_sparse_inference);

  static bool canLoadDatasetInMemory(const std::string& filename);

 private:
  static std::shared_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  loadStreamingDataset(
      const std::string& filename,
      const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
          batch_processor,
      uint32_t batch_size = 128);

  static uint32_t getHiddenLayerSize(const std::string& model_size,
                                     uint64_t n_classes, uint64_t input_dim);

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
  DistributedTrainingContextptr _distributed_train_context;
};

}  // namespace thirdai::bolt