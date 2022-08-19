#pragma once

#include <cereal/archives/binary.hpp>
#include "AutoClassifierBase.h"
#include <bolt/src/graph/Graph.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/TabularBlocks.h>
#include <dataset/src/encodings/categorical/StringCategoricalEncoding.h>
#include <dataset/src/utils/SafeFileIO.h>

namespace thirdai::bolt {

class TabularClassifier {
 public:
  TabularClassifier(const std::string& model_size, uint32_t n_classes)
      : _input_dim(100000), _n_classes(n_classes), _metadata(nullptr) {
    _classifier = std::make_unique<AutoClassifierBase>(
        /* input_dim = */ _input_dim,
        /* n_classes = */ _n_classes, model_size);
  }

  void initClassifierDistributedTraining(
      const std::string& filename, std::vector<std::string>& column_datatypes,
      uint32_t epochs, float learning_rate) {
    if (_metadata) {
      std::cout << "Note: Metadata from the training dataset is used for "
                   "predictions on future test data. Calling train(..) again "
                   "resets this metadata."
                << std::endl;
    }
    _metadata = processTabularMetadata(filename, column_datatypes, 128);

    std::shared_ptr<dataset::GenericBatchProcessor> batch_processor =
        makeTabularBatchProcessor();
    _classifier->initClassifierDistributedTraining(
        filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            batch_processor),
        epochs, learning_rate);
  }

  DistributedTrainingContext getDistributedTrainingContext() {
    return _classifier->getDistributedTrainingContext();
  }

  BoltGraph getBoltGraphModel() { return _classifier->getBoltGraphModel(); }

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

    _classifier->train(
        filename,
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

    _classifier->predict(
        filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            batch_processor),
        output_filename, _metadata->getClassIdToNames());
  }

  std::string predictSingle(std::vector<std::string>& values) {
    if (values.size() != _metadata->numColumns() - 1) {
      throw std::invalid_argument(
          "Passed in an input of size " + std::to_string(values.size()) +
          " but needed a vector of size " +
          std::to_string(_metadata->numColumns() - 1) +
          ". predict_single expects a vector of values in the same format as "
          "the original csv but without the label present.");
    }

    std::vector<std::string_view> encodable_values(values.begin(),
                                                   values.end());

    /*
      the batch processor fails if the number of columns mismatches with the
      original format. since we are only creating an input vector here the
      label is not relevant, thus we add some bogus here in the label's column
    */
    encodable_values.insert(encodable_values.begin() + _metadata->getLabelCol(),
                            /* value = */ " ");

    std::shared_ptr<dataset::GenericBatchProcessor> batch_processor =
        makeTabularBatchProcessor();

    BoltVector input;
    if (auto err = batch_processor->makeInputVector(encodable_values, input)) {
      std::rethrow_exception(err);
    }

    BoltVector output =
        _classifier->predictSingle({input}, {},
                                   /* use_sparse_inference = */ true);

    return _metadata->getClassIdToNames()[output.getIdWithHighestActivation()];
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

    std::shared_ptr<dataset::TabularMetadataProcessor>
        metadata_batch_processor =
            std::make_shared<dataset::TabularMetadataProcessor>(
                column_datatypes, _n_classes);

    // TabularMetadataProcessor inherets ComputeBatchProcessor so this doesn't
    // produce any vectors, we are just using it to iterate over the dataset.
    auto compute_dataset =
        std::make_shared<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
            data_loader, metadata_batch_processor);
    while (compute_dataset->nextBatchTuple()) {
    }

    return metadata_batch_processor->getMetadata();
  }

  std::shared_ptr<dataset::GenericBatchProcessor> makeTabularBatchProcessor() {
    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {
        std::make_shared<dataset::TabularPairGram>(_metadata, _input_dim)};
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
    archive(_input_dim, _n_classes, _metadata, _classifier);
  }

  uint32_t _input_dim;
  uint32_t _n_classes;
  std::shared_ptr<dataset::TabularMetadata> _metadata;
  std::unique_ptr<AutoClassifierBase> _classifier;
};

}  // namespace thirdai::bolt