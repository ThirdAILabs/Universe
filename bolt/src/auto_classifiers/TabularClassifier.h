#pragma once

#include <cereal/archives/binary.hpp>
#include "AutoClassifierUtils.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/TabularBlocks.h>
#include <dataset/src/bolt_datasets/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/bolt_datasets/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/encodings/categorical/StringCategoricalEncoding.h>
#include <dataset/src/utils/SafeFileIO.h>

namespace thirdai::bolt {

class TabularClassifier {
 public:
  TabularClassifier(const std::string& model_size, uint32_t n_classes)
      : _metadata(nullptr) {
    _model = AutoClassifierUtils::createNetwork(/* input_dim = */ 100000,
                                                /* n_classes = */ n_classes,
                                                model_size);
  }

  void train(const std::string& filename,
             std::vector<std::string>& column_datatypes, uint32_t epochs,
             float learning_rate) {
    if (_metadata) {
      std::cout << "Note: Metadata from the training dataset is used for "
                   "predictions on future test data. Calling train(..) again "
                   "resets this metadata."
                << std::endl;
    }
    _metadata = processTabularMetadata(filename, column_datatypes);

    std::shared_ptr<dataset::GenericBatchProcessor> batch_processor =
        makeTabularBatchProcessor();

    AutoClassifierUtils::train(
        _model, filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            batch_processor),
        /* epochs = */ epochs,
        /* learning_rate = */ learning_rate);
  }

  void predict(const std::string& filename,
               const std::optional<std::string>& output_filename) {
    if (!_metadata) {
      throw std::invalid_argument(
          "Cannot call predict(..) without calling train(..) first.");
    }

    std::shared_ptr<dataset::GenericBatchProcessor> batch_processor =
        makeTabularBatchProcessor();

    AutoClassifierUtils::predict(
        _model, filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            batch_processor),
        output_filename, _metadata->getClassIdToNames());
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<TabularClassifier> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<TabularClassifier> deserialize_into(
        new TabularClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
  std::shared_ptr<dataset::TabularMetadata> processTabularMetadata(
      const std::string& filename, std::vector<std::string>& column_datatypes,
      uint32_t batch_size = 256) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

    std::shared_ptr<dataset::TabularMetadataProcessor> batch_processor =
        std::make_shared<dataset::TabularMetadataProcessor>(
            column_datatypes, _model->getOutputDim());

    // TabularMetadataProcessor inherets ComputeBatchProcessor so this doesn't
    // produce any vectors, we are just using it to iterate over the dataset.
    auto compute_dataset =
        std::make_shared<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
            data_loader, batch_processor);
    while (compute_dataset->nextBatchTuple()) {
    }

    return batch_processor->getMetadata();
  }

  std::shared_ptr<dataset::GenericBatchProcessor> makeTabularBatchProcessor() {
    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {
        std::make_shared<dataset::TabularPairGram>(_metadata,
                                                   _model->getInputDim())};
    std::vector<std::shared_ptr<dataset::Block>> target_blocks = {
        std::make_shared<dataset::CategoricalBlock>(
            _metadata->getLabelCol(),
            std::make_shared<dataset::StringCategoricalEncoding>(
                _metadata->getClassToIdMap()))};

    return std::make_shared<dataset::GenericBatchProcessor>(
        /* input_blocks = */ input_blocks,
        /* target_blocks = */ target_blocks, /* has_header = */ true);
  }

  // Private constructor for cereal
  TabularClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_metadata, _model);
  }

  std::shared_ptr<dataset::TabularMetadata> _metadata;
  std::shared_ptr<FullyConnectedNetwork> _model;
};

}  // namespace thirdai::bolt