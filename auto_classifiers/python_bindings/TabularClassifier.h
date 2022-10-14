#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_classifiers/python_bindings/AutoClassifierBase.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/TabularPairGram.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::python {

/**
 * The TabularClassifier takes in tabular data and applies binning + pairgrams
 * to featureize it. The column datatypes list indicates how to bin/featurize
 * the different parts of the dataset and automatically maps the class names
 * to output neurons. Evaluate and predict return lists of strings of the
 * predicted class names.
 */
class TabularClassifier final
    : public AutoClassifierBase<std::vector<std::string>> {
 public:
  TabularClassifier(uint32_t internal_model_dim, uint32_t n_classes,
                    std::vector<std::string> column_datatypes)
      : AutoClassifierBase(
            createAutotunedModel(
                internal_model_dim, n_classes,
                /* sparsity= */ std::nullopt,
                /* output_activation= */ ActivationFunction::Softmax),
            ReturnMode::ClassName),
        _metadata(nullptr),
        _batch_processor(nullptr),
        _column_datatypes(std::move(column_datatypes)) {}

  TabularClassifier(uint32_t internal_model_dim, uint32_t n_classes,
                    std::shared_ptr<dataset::TabularMetadata> metadata)
      : AutoClassifierBase(
            createAutotunedModel(
                internal_model_dim, n_classes,
                /* sparsity= */ std::nullopt,
                /* output_activation= */ ActivationFunction::Softmax),
            ReturnMode::ClassName),
        _metadata(std::move(metadata)),
        _batch_processor(nullptr) {}

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

    if (deserialize_into->_metadata) {
      deserialize_into->createBatchProcessor();
    }

    return deserialize_into;
  }

 protected:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTrainingDataset(std::shared_ptr<dataset::DataLoader> data_loader,
                     std::optional<uint64_t> max_in_memory_batches) final {
    if (!_metadata) {
      processTabularMetadata(data_loader, max_in_memory_batches);
    }

    createBatchProcessor();

    data_loader->restart();

    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), _batch_processor);
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getEvalDataset(std::shared_ptr<dataset::DataLoader> data_loader) final {
    if (!_batch_processor) {
      throw std::runtime_error(
          "Cannot call evaulate on TabularClassifier before calling train.");
    }
    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), _batch_processor);
  }

  BoltVector featurizeInputForInference(
      const std::vector<std::string>& values) final {
    if (!_batch_processor) {
      throw std::runtime_error(
          "Cannot call featurizeInputForInference on TabularClasssifier before "
          "training.");
    }
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

    BoltVector input;
    if (auto err = _batch_processor->makeInputVector(encodable_values, input)) {
      std::rethrow_exception(err);
    }

    return input;
  }

  std::string getClassName(uint32_t neuron_id) final {
    return _metadata->getClassToIdMap()->getString(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 256; }

  std::optional<uint32_t> defaultRebuildHashTablesInterval() const final {
    return 10000;
  }

  std::optional<uint32_t> defaultReconstructHashFunctionsInterval()
      const final {
    return 50000;
  }

  bool freezeHashTablesAfterFirstEpoch() const final { return true; }

  bool useSparseInference() const final { return true; }

  std::vector<std::string> getEvaluationMetrics() const final {
    return {"categorical_accuracy"};
  }

 private:
  void createBatchProcessor() {
    if (!_metadata) {
      throw std::runtime_error(
          "Cannot call createBatchProcessor for tabular classifier with "
          "metadata as nullptr.");
    }
    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {
        std::make_shared<dataset::TabularPairGram>(
            _metadata, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)};

    std::vector<std::shared_ptr<dataset::Block>> target_blocks = {
        dataset::StringLookupCategoricalBlock::make(
            _metadata->getLabelCol(), _metadata->getClassToIdMap())};

    _batch_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks = */ input_blocks,
        /* label_blocks = */ target_blocks, /* has_header = */ true);
  }

  void processTabularMetadata(
      const std::shared_ptr<dataset::DataLoader>& data_loader,
      std::optional<uint64_t> max_in_memory_batches) {
    std::shared_ptr<dataset::TabularMetadataProcessor>
        metadata_batch_processor =
            std::make_shared<dataset::TabularMetadataProcessor>(
                _column_datatypes, _model->outputDim());

    // TabularMetadataProcessor inherets ComputeBatchProcessor so this doesn't
    // produce any vectors, we are just using it to iterate over the dataset.
    auto compute_dataset =
        std::make_shared<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
            data_loader, metadata_batch_processor);

    uint32_t batches_to_use =
        max_in_memory_batches.value_or(std::numeric_limits<uint32_t>::max());
    uint32_t batch_cnt = 0;
    while (compute_dataset->nextBatchTuple() &&
           (batch_cnt++ < batches_to_use)) {
    }

    _metadata = metadata_batch_processor->getTabularMetadata();
  }

  // Private constructor for cereal.
  TabularClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray),
        _metadata(nullptr),
        _batch_processor(nullptr) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this), _metadata,
            _column_datatypes);
  }

  dataset::TabularMetadataPtr _metadata;
  dataset::GenericBatchProcessorPtr _batch_processor;
  std::vector<std::string> _column_datatypes;
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::TabularClassifier)
