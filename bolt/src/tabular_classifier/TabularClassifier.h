#pragma once

#include <cereal/archives/binary.hpp>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/blocks/TabularBlocks.h>
#include <dataset/src/bolt_datasets/batch_processors/TabularBatchProcessor.h>
#include <dataset/src/bolt_datasets/batch_processors/TabularMetadataProcessor.h>
#include <memory>

namespace thirdai::bolt {

class TabularClassifier {
 public:
  TabularClassifier(const std::string& model_size, uint32_t n_classes);

  void train(const std::string& filename,
             std::vector<std::string>& column_datatypes, uint32_t epochs,
             float learning_rate);

  void predict(const std::string& filename,
               const std::optional<std::string>& output_filename);

  void save(const std::string& filename) {
    std::ofstream filestream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<TabularClassifier> load(const std::string& filename) {
    std::ifstream filestream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<TabularClassifier> deserialize_into(
        new TabularClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
  dataset::TabularMetadata getTabularMetadata(
      const std::string& filename, std::vector<std::string>& column_datatypes,
      uint32_t batch_size = 256) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

    std::shared_ptr<dataset::TabularMetadataProcessor> batch_processor =
        std::make_shared<dataset::TabularMetadataProcessor>(column_datatypes);

    std::make_shared<dataset::StreamingDataset<BoltBatch>>(data_loader,
                                                           batch_processor);

    return batch_processor.getMetadata();
  }

  std::shared_ptr<dataset::StreamingDataset<BoltBatch>> loadStreamingDataset(
      const std::string& filename, dataset::TabularMetadata metadata,
      uint32_t batch_size = 256) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {
        std::make_shared<dataset::TabularPairGram>(metadata, _input_dim)};
    std::vector<std::shared_ptr<dataset::Block>> target_blocks = {
        std::make_shared<dataset::TabularLabel>(metadata)};

    std::shared_ptr<dataset::TabularBatchProcessor> batch_processor =
        std::make_shared<dataset::TabularBatchProcessor>(
            /* input_blocks */ input_blocks, /* target_blocks */ target_blocks);

    auto dataset = std::make_shared<dataset::StreamingDataset<BoltBatch>>(
        data_loader, batch_processor);
    return dataset;
  }

  // Private constructor for cereal
  TabularClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model);
  }

  uint32_t _input_dim;
  dataset::TabularMetadata
      _metadata;  // TODO(david) make shared pointer/optional??
  std::unique_ptr<FullyConnectedNetwork> _model;
};

}  // namespace thirdai::bolt