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
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt::python {

class TextClassifier final : public AutoClassifierBase {
 public:
  TextClassifier(uint32_t hidden_layer_dim, uint32_t n_classes)
      : AutoClassifierBase(createModel(hidden_layer_dim, n_classes),
                           ReturnMode::ClassName) {
    auto label_block =
        dataset::StringLookupCategoricalBlock::make(/* col= */ 0, _n_classes);
    _label_id_lookup = label_block->getVocabulary();

    _batch_processor = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)}, {label_block});
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

    deserialize_into->reinitializeBatchProcessors();

    return deserialize_into;
  }

 protected:
  dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) final {
    (void)data_loader;
    (void)max_in_memory_batches;
    return _batch_processor;
  }

  dataset::GenericBatchProcessorPtr getPredictBatchProcessor() final {
    return _batch_processor;
  }

  BoltVector featurizeInputForInference(const py::object& input) final {
    // TODO(Nicholas): check type:
    std::string input_str = input.cast<std::string>();

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
  static BoltGraphPtr createModel(uint32_t hidden_layer_dim,
                                  uint32_t n_classes) {
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

  void reinitializeBatchProcessors() {
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        /* col= */ 0, _label_id_lookup);
    _batch_processor = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)}, {label_block});
  }

  // Private constructor for cereal.
  TextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray),
        _label_id_lookup(nullptr) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this), _label_id_lookup);
  }

  dataset::GenericBatchProcessorPtr _batch_processor;
  dataset::ThreadSafeVocabularyPtr _label_id_lookup;
  uint32_t _n_classes;
};

class MultiLabelTextClassifier final : public AutoClassifierBase {
 public:
  explicit MultiLabelTextClassifier(uint32_t n_classes, float threshold = 0.95)
      : AutoClassifierBase(createModel(n_classes),
                           ReturnMode::NumpyArrayWithThresholding, threshold),
        _n_classes(n_classes) {
    buildBatchProcessors();
  }

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

    deserialize_into->buildBatchProcessors();

    return deserialize_into;
  }

 protected:
  dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) final {
    (void)data_loader;
    (void)max_in_memory_batches;
    return _batch_processor;
  }

  dataset::GenericBatchProcessorPtr getPredictBatchProcessor() final {
    return _batch_processor;
  }

  BoltVector featurizeInputForInference(const py::object& input) final {
    // TODO(Nicholas): Check input type
    std::string sentence =
        tokensToSentence(input.cast<std::vector<uint32_t>>());

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

  void buildBatchProcessors() {
    _batch_processor = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)},
        {dataset::NumericalCategoricalBlock::make(/* col= */ 0,
                                                  /* n_classes= */ _n_classes)},
        /* has_header= */ false, /* delimiter= */ '\t');

    _inference_featurizer = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 0)}, {},
        /* has_header= */ false, /* delimiter= */ '\t');
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
    archive(cereal::base_class<AutoClassifierBase>(this), _n_classes);
  }

  dataset::GenericBatchProcessorPtr _batch_processor;
  dataset::GenericBatchProcessorPtr _inference_featurizer;
  uint32_t _n_classes;
};

class TabularClassifier final : public AutoClassifierBase {
 protected:
  dataset::GenericBatchProcessorPtr getTrainingBatchProcessor(
      std::shared_ptr<dataset::DataLoader> data_loader,
      std::optional<uint64_t> max_in_memory_batches) final;

  dataset::GenericBatchProcessorPtr getPredictBatchProcessor() final;

  BoltVector featurizeInputForInference(const py::object& input) final;

  std::string getClassName(uint32_t neuron_id) final;

  uint32_t defaultBatchSize() const final;

  bool freezeHashTables() const final;

  bool useSparseInference() const final;

  std::vector<std::string> getPredictMetrics() const final;
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::TextClassifier)
CEREAL_REGISTER_TYPE(thirdai::bolt::python::MultiLabelTextClassifier)