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
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/TabularBlocks.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::python {

inline BoltGraphPtr createAutotunedModel(uint32_t internal_model_dim,
                                         uint32_t n_classes,
                                         std::optional<float> sparsity,
                                         ActivationFunction output_activation);
inline std::string joinTokensIntoString(const std::vector<uint32_t>& tokens,
                                        char delimiter);
inline float autotunedHiddenLayerSparsity(uint64_t layer_dim);

/**
 * The TextClassifier takes in data in the form:
 *        <class_name>,<text>.
 * It uses paigrams to featurize the text and automatically maps the class names
 * to output neurons. Evaluate and predict return lists of strings of the
 * predicted class names.
 */
class TextClassifier final : public AutoClassifierBase<std::string> {
 public:
  TextClassifier(uint32_t internal_model_dim, uint32_t n_classes)
      : AutoClassifierBase(
            createAutotunedModel(
                internal_model_dim, n_classes,
                /* sparsity= */ std::nullopt,
                /* output_activation= */ ActivationFunction::Softmax),
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
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTrainingDataset(std::shared_ptr<dataset::DataLoader> data_loader,
                     std::optional<uint64_t> max_in_memory_batches) final {
    (void)max_in_memory_batches;
    return getDataset(data_loader);
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getEvalDataset(std::shared_ptr<dataset::DataLoader> data_loader) final {
    return getDataset(data_loader);
  }

  BoltVector featurizeInputForInference(const std::string& input_str) final {
    return dataset::TextEncodingUtils::computePairgrams(
        input_str, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    return _label_id_lookup->getString(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 256; }

  bool freezeHashTablesAfterFirstEpoch() const final { return true; }

  bool useSparseInference() const final { return true; }

  std::vector<std::string> getEvaluationMetrics() const final {
    return {"categorical_accuracy"};
  }

 private:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>> getDataset(
      std::shared_ptr<dataset::DataLoader> data_loader) {
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        /* col= */ 0, _label_id_lookup);
    auto batch_processor = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)}, {label_block});

    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), batch_processor);
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

  dataset::ThreadSafeVocabularyPtr _label_id_lookup;
};

/**
 * The MultiLabelTextClassifier takes in data in the form:
 *        <class_id_1>,<class_id_2>,...,<class_id_n>\t<text>.
 * It uses paigrams to featurize the text, and uses sigmoid/bce to handle the
 * variable number of labels. Thresholding is applied to ensure that each
 * prediction has at least one neuron with an activation > the given threshold.
 * Predict and evaluate return numpy arrays of the output activations.
 */
class MultiLabelTextClassifier final
    : public AutoClassifierBase<std::vector<uint32_t>> {
 public:
  explicit MultiLabelTextClassifier(uint32_t n_classes, float threshold = 0.95)
      : AutoClassifierBase(createModel(n_classes), ReturnMode::NumpyArray),
        _threshold(threshold) {}

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

  void updateThreshold(float new_threshold) { _threshold = new_threshold; }

 protected:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTrainingDataset(std::shared_ptr<dataset::DataLoader> data_loader,
                     std::optional<uint64_t> max_in_memory_batches) final {
    (void)max_in_memory_batches;
    return getDataset(data_loader);
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getEvalDataset(std::shared_ptr<dataset::DataLoader> data_loader) final {
    return getDataset(data_loader);
  }

  void processPredictionBeforeReturning(uint32_t* active_neurons,
                                        float* activations,
                                        uint32_t len) final {
    (void)active_neurons;

    uint32_t max_id = getMaxIndex(activations, len);
    if (activations[max_id] < _threshold) {
      activations[max_id] = _threshold + 0.0001;
    }
  }

  BoltVector featurizeInputForInference(
      const std::vector<uint32_t>& input) final {
    std::string sentence = joinTokensIntoString(input, /* delimiter= */ ' ');

    return dataset::TextEncodingUtils::computePairgrams(
        sentence, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    return std::to_string(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 2048; }

  bool freezeHashTablesAfterFirstEpoch() const final { return false; }

  bool useSparseInference() const final { return false; }

  std::vector<std::string> getEvaluationMetrics() const final {
    std::string f_measure = "f_measure(" + std::to_string(_threshold) + ")";
    return {"categorical_accuracy", f_measure};
  }

 private:
  static BoltGraphPtr createModel(uint32_t n_classes) {
    auto model = CommonNetworks::FullyConnected(
        /* input_dim= */ dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
        /* layers= */ {FullyConnectedNode::makeDense(
                           /* dim= */ 1024, "relu"),
                       FullyConnectedNode::makeExplicitSamplingConfig(
                           /* dim= */ n_classes,
                           /* sparsity= */ getOutputSparsity(n_classes),
                           /* activation= */ "sigmoid",
                           /* num_tables= */ 64, /* hashes_per_table= */ 4,
                           /* reservoir_size= */ 64)});
    model->compile(std::make_shared<BinaryCrossEntropyLoss>(),
                   /* print_when_done= */ false);

    return model;
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>> getDataset(
      std::shared_ptr<dataset::DataLoader> data_loader) {
    auto batch_processor = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)},
        {dataset::NumericalCategoricalBlock::make(
            /* col= */ 0,
            /* n_classes= */ _model->outputDim(), /* delimiter= */ ',')},
        /* has_header= */ false, /* delimiter= */ '\t');

    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), batch_processor);
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

  float _threshold;

  // Private constructor for cereal.
  MultiLabelTextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this), _threshold);
  }
};

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
        _classname_to_id_lookup(nullptr),
        _metadata(nullptr),
        _batch_processor(nullptr),
        _column_datatypes(std::move(column_datatypes)) {}

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
    processTabularMetadata(data_loader, max_in_memory_batches);

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
    return _classname_to_id_lookup->getString(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 256; }

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

    _classname_to_id_lookup =
        dataset::ThreadSafeVocabulary::make(_metadata->getClassToIdMap(),
                                            /* fixed= */ true);

    std::vector<std::shared_ptr<dataset::Block>> target_blocks = {
        dataset::StringLookupCategoricalBlock::make(_metadata->getLabelCol(),
                                                    _classname_to_id_lookup)};

    _batch_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks = */ input_blocks,
        /* label_blocks = */ target_blocks, /* has_header = */ true);
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

    uint32_t batches_to_use =
        max_in_memory_batches.value_or(std::numeric_limits<uint32_t>::max());
    uint32_t batch_cnt = 0;
    while (compute_dataset->nextBatchTuple() &&
           (batch_cnt++ < batches_to_use)) {
    }

    _metadata = metadata_batch_processor->getMetadata();
  }

  // Private constructor for cereal.
  TabularClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray),
        _classname_to_id_lookup(nullptr),
        _metadata(nullptr),
        _batch_processor(nullptr) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this),
            _classname_to_id_lookup, _metadata, _column_datatypes);
  }

  dataset::ThreadSafeVocabularyPtr _classname_to_id_lookup;
  std::shared_ptr<dataset::TabularMetadata> _metadata;
  dataset::GenericBatchProcessorPtr _batch_processor;
  std::vector<std::string> _column_datatypes;
};

class BinaryTextClassifier final
    : public AutoClassifierBase<std::vector<uint32_t>> {
 public:
  explicit BinaryTextClassifier(uint32_t n_outputs, uint32_t internal_model_dim,
                                std::optional<float> sparsity = std::nullopt,
                                bool use_sparse_inference = true)
      : AutoClassifierBase(
            createAutotunedModel(
                /* internal_model_dim= */ internal_model_dim, n_outputs,
                sparsity,
                /* output_activation= */ ActivationFunction::Sigmoid),
            ReturnMode::NumpyArray),
        _use_sparse_inference(use_sparse_inference) {}

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<BinaryTextClassifier> load(
      const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<BinaryTextClassifier> deserialize_into(
        new BinaryTextClassifier());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 protected:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTrainingDataset(std::shared_ptr<dataset::DataLoader> data_loader,
                     std::optional<uint64_t> max_in_memory_batches) final {
    (void)max_in_memory_batches;
    return getDataset(data_loader);
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getEvalDataset(std::shared_ptr<dataset::DataLoader> data_loader) final {
    return getDataset(data_loader);
  }

  BoltVector featurizeInputForInference(
      const std::vector<uint32_t>& tokens) final {
    std::string sentence = joinTokensIntoString(tokens, /* delimiter= */ ' ');

    return dataset::TextEncodingUtils::computeUnigrams(
        sentence, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    (void)neuron_id;
    throw std::runtime_error(
        "getClassName() is not support for BinaryTextClassifier.");
  }

  uint32_t defaultBatchSize() const final { return 256; }

  bool freezeHashTablesAfterFirstEpoch() const final {
    return _use_sparse_inference;
  }

  bool useSparseInference() const final { return _use_sparse_inference; }

  std::vector<std::string> getEvaluationMetrics() const final { return {}; }

 private:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>> getDataset(
      std::shared_ptr<dataset::DataLoader> data_loader) {
    // Because we have n_outputs binary label columns, the text column is starts
    // at _model->outputDim() which is equivalent to n_classes.
    auto batch_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ {dataset::UniGramTextBlock::make(
            /* col= */ _model->outputDim())},
        /* label_blocks= */ {
            dataset::DenseArrayBlock::make(/* start_col= */ 0,
                                           /* dim= */ _model->outputDim())});

    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), batch_processor);
  }

  // Private constructor for cereal.
  BinaryTextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this));
  }

  bool _use_sparse_inference;
};

inline BoltGraphPtr createAutotunedModel(
    uint32_t internal_model_dim, uint32_t n_classes,
    std::optional<float> hidden_layer_sparsity,
    ActivationFunction output_activation) {
  auto input_layer =
      Input::make(dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);

  auto hidden_layer = FullyConnectedNode::makeAutotuned(
      /* dim= */ internal_model_dim,
      /* sparsity= */
      hidden_layer_sparsity.value_or(
          autotunedHiddenLayerSparsity(internal_model_dim)),
      /* activation= */ "relu");
  hidden_layer->addPredecessor(input_layer);

  FullyConnectedNodePtr output_layer;
  std::shared_ptr<LossFunction> loss;

  if (output_activation == ActivationFunction::Softmax) {
    output_layer = FullyConnectedNode::makeDense(
        /* dim= */ n_classes,
        /* activation= */ "softmax");
    loss = std::make_shared<CategoricalCrossEntropyLoss>();
  } else if (output_activation == ActivationFunction::Sigmoid) {
    loss = std::make_shared<BinaryCrossEntropyLoss>();
    output_layer = FullyConnectedNode::makeDense(
        /* dim= */ n_classes,
        /* activation= */ "sigmoid");
  } else {
    throw std::invalid_argument(
        "Output activation in createAutotunedModel must be Softmax or "
        "Sigmoid.");
  }

  output_layer->addPredecessor(hidden_layer);

  auto model = std::make_shared<BoltGraph>(std::vector<InputPtr>{input_layer},
                                           output_layer);

  model->compile(loss, /* print_when_done= */ false);

  return model;
}

inline std::string joinTokensIntoString(const std::vector<uint32_t>& tokens,
                                        char delimiter) {
  std::stringstream sentence_ss;
  for (uint32_t i = 0; i < tokens.size(); i++) {
    if (i > 0) {
      sentence_ss << delimiter;
    }
    sentence_ss << tokens[i];
  }
  return sentence_ss.str();
}

inline float autotunedHiddenLayerSparsity(uint64_t layer_dim) {
  if (layer_dim < 300) {
    return 1.0;
  }
  if (layer_dim < 1500) {
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
CEREAL_REGISTER_TYPE(thirdai::bolt::python::BinaryTextClassifier)