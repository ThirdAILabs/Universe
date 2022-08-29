#pragma once

#include <bolt/src/auto_classifiers/AutoClassifierBase.h>
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

class MultiLabelTextClassifier {
 public:
  explicit MultiLabelTextClassifier(uint32_t n_classes)
      : _n_classes(n_classes) {
    buildBatchProcessors(n_classes);

    assert(n_classes == _labeled_processor->getLabelDim());

    /*
      TODO(Geordie, Tharun): Add more tuned presets / make configurable by size.
      While these numbers are currently hard-coded for Wayfair, they are also
      reasonable configurations in general.
    */
    _classifier = CommonNetworks::FullyConnected(
        /* input_dim= */ _labeled_processor->getInputDim(),
        /* layers= */ {FullyConnectedNode::make(
                           /* dim= */ 1024, "relu"),
                       FullyConnectedNode::make(
                           /* dim= */ n_classes,
                           /* sparsity= */ getOutputSparsity(n_classes),
                           /* activation= */ "sigmoid",
                           /* num_tables= */ 64, /* hashes_per_table= */ 4,
                           /* reservoir_size= */ 64)});
    _classifier->compile(std::make_shared<BinaryCrossEntropyLoss>(),
                         /* print_when_done= */ false);
  }

  void train(const std::string& filename, uint32_t epochs, float learning_rate,
             const std::vector<std::string>& metrics = {}) {
    dataset::StreamingGenericDatasetLoader dataset(filename, _labeled_processor,
                                                   /* batch_size= */ 2048,
                                                   /* shuffle= */ true);

    if (!AutoClassifierBase::canLoadDatasetInMemory(filename)) {
      throw std::invalid_argument("Cannot load training dataset in memory.");
    }

    auto [train_data, train_labels] = dataset.loadInMemory();

    auto config = TrainConfig::makeConfig(learning_rate, epochs)
                      .withMetrics(metrics)
                      .withRebuildHashTables(10000)
                      .withReconstructHashFunctions(50000);

    _classifier->train({train_data}, {}, train_labels, config);
  }

  InferenceResult predict(const std::string& filename,
                          const std::vector<std::string>& metrics = {}) {
    dataset::StreamingGenericDatasetLoader dataset(filename, _labeled_processor,
                                                   /* batch_size= */ 2048);

    if (!AutoClassifierBase::canLoadDatasetInMemory(filename)) {
      throw std::invalid_argument("Cannot load prediction dataset in memory.");
    }

    auto [pred_data, pred_labels] = dataset.loadInMemory();

    auto config = PredictConfig::makeConfig().withMetrics(metrics);

    return _classifier->predict({pred_data}, {}, pred_labels, config);
  }

  BoltVector predictSingleFromSentence(std::string sentence,
                                       float threshold = 0.95) {
    float epsilon = 0.001;

    // The following step must be separate from the above
    // because we need to keep the sentence in scope and alive.
    std::vector<std::string_view> sample = {
        std::string_view(sentence.data(), sentence.size())};

    BoltVector input_vector;
    if (auto exception =
            _unlabeled_processor->makeInputVector(sample, input_vector)) {
      std::rethrow_exception(exception);
    }

    BoltVector output =
        _classifier->predictSingle({input_vector}, {},
                                   /* use_sparse_inference = */ false);

    assert(output.isDense());
    auto max_id = output.getHighestActivationId();
    if (output.activations[max_id] < threshold) {
      output.activations[max_id] = threshold + epsilon;
    }

    return output;
  }

  /**
   * We provide a convenience function to predict with tokens because
   * we are ware that practitioners may preprocess data with their own
   * tokenizers.
   *
   * We don't have a similar convenience function for training and
   * predicting because even if they do use tokens, we will read
   * these tokens from a file anyway so there is no need for such
   * a function.
   */
  BoltVector predictSingleFromTokens(const std::vector<uint32_t>& tokens,
                                     float threshold = 0.95) {
    std::string sentence = tokensToSentence(tokens);
    return predictSingleFromSentence(sentence, threshold);
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
    deserialize_into->buildBatchProcessors(deserialize_into->_n_classes);
    return deserialize_into;
  }

  uint32_t numClasses() const { return _n_classes; }

 private:
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

  void buildBatchProcessors(uint32_t n_classes) {
    _labeled_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ {dataset::PairGramTextBlock::make(/* col= */ 1)},
        /* label_blocks= */
        {dataset::NumericalCategoricalBlock::make(
            /* col= */ 0, /* n_classes= */ n_classes, /* delimiter= */ ',')},
        /* has_header= */ false, /* delimiter= */ '\t');

    _unlabeled_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ {dataset::PairGramTextBlock::make(/* col= */ 0)},
        /* label_blocks= */ {},
        /* has_header= */ false, /* delimiter= */ '\t');
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

  // Private constructor for cereal
  MultiLabelTextClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_n_classes, _classifier);
  }

  uint32_t _n_classes;
  std::shared_ptr<dataset::GenericBatchProcessor> _labeled_processor;
  std::shared_ptr<dataset::GenericBatchProcessor> _unlabeled_processor;
  BoltGraphPtr _classifier;
};

}  // namespace thirdai::bolt