#pragma once

#include <cereal/archives/binary.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <bolt/src/graph/Graph.h>
#include <memory>

namespace thirdai::bolt {

class MultiLabelTextClassifier {
 public:
  explicit MultiLabelTextClassifier(uint32_t n_classes);

  void train(const std::string& filename, uint32_t epochs, float learning_rate);

  void predict(const std::string& filename, const std::optional<std::string>& output_filename, float threshold = 0.8);

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<MultiLabelTextClassifier> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<MultiLabelTextClassifier> deserialize_into(new MultiLabelTextClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
 std::shared_ptr<dataset::StreamingDataset<BoltBatch>> loadStreamingDataset(
      const std::string& filename, uint32_t batch_size = 256) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

    auto dataset = std::make_shared<dataset::StreamingDataset<BoltBatch>>(
        data_loader, _batch_processor);
    return dataset;
  }

  // Private constructor for cereal
  MultiLabelTextClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _batch_processor);
  }

  std::shared_ptr<BoltGraph> _model;
  std::shared_ptr<dataset::GenericBatchProcessor> _batch_processor;
};

}  // namespace thirdai::bolt