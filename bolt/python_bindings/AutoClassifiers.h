#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/python_bindings/AutoClassifierBase.h>
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/TabularBlocks.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <limits>

namespace thirdai::bolt::python {

inline BoltGraphPtr createModel(uint32_t hidden_layer_dim, uint32_t n_classes);
inline float getHiddenLayerSparsity(uint64_t layer_dim);

class TextClassifier final : public AutoClassifierBase<const std::string&> {
 public:
  TextClassifier(uint32_t hidden_layer_dim, uint32_t n_classes)
      : AutoClassifierBase(createModel(hidden_layer_dim, n_classes),
                           ReturnMode::ClassName) {
    _label_id_lookup = dataset::ThreadSafeVocabulary::make(n_classes);
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<TextClassifier> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<TextClassifier> deserialize_into(new TextClassifier());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 protected:
  dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) final {
    (void)data_loader;
    (void)max_in_memory_batches;
    return getPredictBatchProcessor();
  }

  dataset::GenericBatchProcessorPtr getPredictBatchProcessor() final {
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        /* col= */ 0, _label_id_lookup);
    return dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)}, {label_block});
  }

  BoltVector featurizeInputForInference(const std::string& input_str) final {
    return dataset::TextEncodingUtils::computePairgrams(
        input_str, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    return _label_id_lookup->getString(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 256; }

  bool freezeHashTables() const final { return true; }

  bool useSparseInference() const final { return true; }

  std::vector<std::string> getPredictMetrics() const final {
    return {"categorical_accuracy"};
  }

 private:
  // Private constructor for cereal.
  TextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray),
        _label_id_lookup(nullptr) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this), _label_id_lookup);
  }

  dataset::ThreadSafeVocabularyPtr _label_id_lookup;
};

class MultiLabelTextClassifier final
    : public AutoClassifierBase<const std::vector<uint32_t>&> {
 public:
  explicit MultiLabelTextClassifier(uint32_t n_classes, float threshold = 0.95)
      : AutoClassifierBase(createModel(n_classes),
                           ReturnMode::NumpyArrayWithThresholding, threshold) {}

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<MultiLabelTextClassifier> load(
      const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<MultiLabelTextClassifier> deserialize_into(
        new MultiLabelTextClassifier());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 protected:
  dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) final {
    (void)data_loader;
    (void)max_in_memory_batches;
    return getPredictBatchProcessor();
  }

  dataset::GenericBatchProcessorPtr getPredictBatchProcessor() final {
    return dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)},
        {dataset::NumericalCategoricalBlock::make(
            /* col= */ 0,
            /* n_classes= */ _model->outputDim())},
        /* has_header= */ false, /* delimiter= */ '\t');
  }

  BoltVector featurizeInputForInference(
      const std::vector<uint32_t>& input) final {
    std::string sentence = tokensToSentence(input);

    return dataset::TextEncodingUtils::computePairgrams(
        sentence, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    return std::to_string(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 2048; }

  bool freezeHashTables() const final { return false; }

  bool useSparseInference() const final { return false; }

  std::vector<std::string> getPredictMetrics() const final {
    return {"categorical_accuracy"};
  }

 private:
  static BoltGraphPtr createModel(uint32_t n_classes) {
    auto model = CommonNetworks::FullyConnected(
        /* input_dim= */ dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
        /* layers= */ {FullyConnectedNode::make(
                           /* dim= */ 1024, "relu"),
                       FullyConnectedNode::make(
                           /* dim= */ n_classes,
                           /* sparsity= */ getOutputSparsity(n_classes),
                           /* activation= */ "sigmoid",
                           /* num_tables= */ 64, /* hashes_per_table= */ 4,
                           /* reservoir_size= */ 64)});
    model->compile(std::make_shared<BinaryCrossEntropyLoss>(),
                   /* print_when_done= */ false);

    return model;
  }

  static float getOutputSparsity(uint32_t output_dim) {
    /*
      For smaller output layers, we return a sparsity
      that puts the sparse dimension between 80 and 160.
    */
    if (output_dim < 450) {
      return 1.0;
    }
    if (output_dim < 900) {
      return 0.2;
    }
    if (output_dim < 1800) {
      return 0.1;
    }
    /*
      For larger layers, we return a sparsity that
      puts the sparse dimension between 100 and 260.
    */
    if (output_dim < 4000) {
      return 0.05;
    }
    if (output_dim < 10000) {
      return 0.02;
    }
    if (output_dim < 20000) {
      return 0.01;
    }
    return 0.05;
  }

  static std::string tokensToSentence(const std::vector<uint32_t>& tokens) {
    std::stringstream sentence_ss;
    for (uint32_t i = 0; i < tokens.size(); i++) {
      if (i > 0) {
        sentence_ss << ' ';
      }
      sentence_ss << tokens[i];
    }
    return sentence_ss.str();
  }

  // Private constructor for cereal.
  MultiLabelTextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this));
  }
};

class TabularClassifier final
    : public AutoClassifierBase<const std::vector<std::string>&> {
 public:
  TabularClassifier(uint32_t hidden_layer_dim, uint32_t n_classes,
                    std::vector<std::string> column_datatypes)
      : AutoClassifierBase(createModel(hidden_layer_dim, n_classes),
                           ReturnMode::NumpyArray),
        _column_datatypes(std::move(column_datatypes)) {}

 protected:
  dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) final {
    processTabularMetadata(data_loader, max_in_memory_batches);

    return getBatchProcessor();
  }

  dataset::GenericBatchProcessorPtr getPredictBatchProcessor() final {
    return getBatchProcessor();
  }

  BoltVector featurizeInputForInference(
      const std::vector<std::string>& values) final {
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

    dataset::GenericBatchProcessorPtr batch_processor = getBatchProcessor();

    BoltVector input;
    if (auto err = batch_processor->makeInputVector(encodable_values, input)) {
      std::rethrow_exception(err);
    }

    return input;
  }

  std::string getClassName(uint32_t neuron_id) final {
    return _vocab->getString(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 256; }

  bool freezeHashTables() const final { return true; }

  bool useSparseInference() const final { return true; }

  std::vector<std::string> getPredictMetrics() const final {
    return {"categorical_accuracy"};
  }

 private:
  dataset::GenericBatchProcessorPtr getBatchProcessor() {
    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {
        std::make_shared<dataset::TabularPairGram>(
            _metadata, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)};

    _vocab = dataset::ThreadSafeVocabulary::make(_metadata->getClassToIdMap(),
                                                 /* fixed= */ true);

    std::vector<std::shared_ptr<dataset::Block>> target_blocks = {
        dataset::StringLookupCategoricalBlock::make(_metadata->getLabelCol(),
                                                    _vocab)};

    return std::make_shared<dataset::GenericBatchProcessor>(
        /* input_blocks = */ input_blocks,
        /* target_blocks = */ target_blocks, /* has_header = */ true);
  }

  void processTabularMetadata(
      const std::shared_ptr<dataset::DataLoader>& data_loader,

      std::optional<uint32_t> max_in_memory_batches) {
    std::shared_ptr<dataset::TabularMetadataProcessor>
        metadata_batch_processor =
            std::make_shared<dataset::TabularMetadataProcessor>(
                _column_datatypes, _model->outputDim());

    // TabularMetadataProcessor inherets ComputeBatchProcessor so this doesn't
    // produce any vectors, we are just using it to iterate over the dataset.
    auto compute_dataset =
        std::make_shared<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
            data_loader, metadata_batch_processor);

    compute_dataset->loadInMemory(
        max_in_memory_batches.value_or(std::numeric_limits<uint64_t>::max()));

    _metadata = metadata_batch_processor->getMetadata();
  }

  dataset::ThreadSafeVocabularyPtr _vocab;
  std::shared_ptr<dataset::TabularMetadata> _metadata;
  std::vector<std::string> _column_datatypes;
};

inline BoltGraphPtr createModel(uint32_t hidden_layer_dim, uint32_t n_classes) {
  auto input_layer = std::make_shared<Input>(
      dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);

  auto hidden_layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ hidden_layer_dim,
      /* sparsity= */ getHiddenLayerSparsity(hidden_layer_dim),
      /* activation= */ "relu");
  hidden_layer->addPredecessor(input_layer);

  auto output_layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ n_classes, /* activation= */ "softmax");
  output_layer->addPredecessor(hidden_layer);

  auto model = std::make_shared<BoltGraph>(std::vector<InputPtr>{input_layer},
                                           output_layer);

  model->compile(std::make_shared<CategoricalCrossEntropyLoss>(),
                 /* print_when_done= */ false);

  return model;
}

inline float getHiddenLayerSparsity(uint64_t layer_dim) {
  if (layer_dim < 300) {
    return 1.0;
  }
  if (layer_dim < 1000) {
    return 0.2;
  }
  if (layer_dim < 4000) {
    return 0.1;
  }
  if (layer_dim < 10000) {
    return 0.05;
  }
  if (layer_dim < 30000) {
    return 0.01;
  }
  return 0.005;
}

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::TextClassifier)
CEREAL_REGISTER_TYPE(thirdai::bolt::python::MultiLabelTextClassifier)
CEREAL_REGISTER_TYPE(thirdai::bolt::python::TabularClassifier)