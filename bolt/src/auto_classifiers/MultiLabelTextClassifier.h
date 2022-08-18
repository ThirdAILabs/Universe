#pragma once

#include <bolt/src/auto_classifiers/AutoClassifierBase.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/encodings/categorical/CategoricalMultiLabel.h>
#include <dataset/src/encodings/text/PairGram.h>
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
  explicit MultiLabelTextClassifier(uint32_t n_classes) : _n_classes(n_classes) {
    buildBatchProcessors(n_classes);

    assert(n_classes == _processor->getLabelDim());

    std::vector<std::pair<uint32_t, float>> hidden_layer_config = {{1024, 1.0}};

    _classifier = CommonNetworks::FullyConnected(
      /* input_dim= */ _processor->getInputDim(),
      /* layers= */ {
        FullyConnectedNode::make(
          /* dim= */ 1024, 
          "relu"),
        FullyConnectedNode::make(
          /* dim= */ n_classes, 
          /* sparsity= */ n_classes >= 500 ? 0.1 : 1, 
          "sigmoid", /* num_tables= */ 64, /* hashes_per_table= */ 4, /* reservoir_size= */ 64)
      }
    );
    _classifier->compile(std::make_shared<BinaryCrossEntropyLoss>(), /* print_when_done= */ false); 

  }

  void train(const std::string& filename, uint32_t epochs, float learning_rate,
             const std::vector<float>& fmeasure_thresholds = {0.9}) {
    std::vector<std::string> metrics;
    for (auto threshold : fmeasure_thresholds) {
      std::stringstream metric_ss;
      metric_ss << "f_measure(" << threshold << ")";
      metrics.push_back(metric_ss.str());
    }

    dataset::StreamingGenericDatasetLoader dataset(
        filename, _processor, /* batch_size= */ 2048, /* shuffle= */ true);

    if (!AutoClassifierBase::canLoadDatasetInMemory(filename)) {
      throw std::invalid_argument("Cannot load training dataset in memory.");
    }

    std::cout << "Loading training dataset from " << filename << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto [train_data, train_labels] = dataset.loadInMemory();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Finished loading the dataset in "
              << std::chrono::duration_cast<std::chrono::seconds>(end_time -
                                                                  start_time)
                     .count()
              << " seconds." << std::endl;

    auto config =
        TrainConfig::makeConfig(learning_rate, epochs)
          .withMetrics(metrics)
          .withRebuildHashTables(10000)
          .withReconstructHashFunctions(50000);

    _classifier->train({train_data}, {}, train_labels, config);
  }

  InferenceResult predict(const std::string& filename,
                          const std::vector<float>& fmeasure_thresholds = {
                              0.9}) {
    std::vector<std::string> metrics;
    for (auto threshold : fmeasure_thresholds) {
      std::stringstream metric_ss;
      metric_ss << "f_measure(" << threshold << ")";
      metrics.push_back(metric_ss.str());
    }

    dataset::StreamingGenericDatasetLoader dataset(filename, _processor,
                                                   /* batch_size= */ 2048);

    if (!AutoClassifierBase::canLoadDatasetInMemory(filename)) {
      throw std::invalid_argument("Cannot load prediction dataset in memory.");
    }

    std::cout << "Loading prediction dataset from " << filename << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto [pred_data, pred_labels] = dataset.loadInMemory();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Finished loading the dataset in "
              << std::chrono::duration_cast<std::chrono::seconds>(end_time -
                                                                  start_time)
                     .count()
              << " seconds." << std::endl;

    auto config = PredictConfig::makeConfig().withMetrics(metrics);

    return _classifier->predict({pred_data}, {}, pred_labels, config);
  }

  BoltVector predictSingle(const std::vector<uint32_t>& tokens,
                           float threshold = 0.9) {
    float epsilon = 0.01;

    std::string sentence = tokensToSentence(tokens);
    // The following step must be separate from the above
    // because we need to keep the sentence in scope and alive.
    std::vector<std::string_view> sample = {std::string_view(sentence.data(), sentence.size())};

    BoltVector input_vector;
    auto exception = _processor->makeInputVector(sample, input_vector);
    if (exception) {
      std::rethrow_exception(exception);
    }

    BoltVector output =
        _classifier->predictSingle({input_vector}, {},
                                   /* use_sparse_inference = */ false);

    assert(output.isDense());
    auto max_id = output.getIdWithHighestActivation();
    if (output.activations[max_id] < threshold) {
      output.activations[max_id] = threshold + epsilon;
    }

    return output;
  }

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
    std::unique_ptr<MultiLabelTextClassifier> deserialize_into(
        new MultiLabelTextClassifier());
    iarchive(*deserialize_into);
    deserialize_into->buildBatchProcessors(deserialize_into->_n_classes);
    return deserialize_into;
  }

 protected:
  void buildBatchProcessors(uint32_t n_classes) {
    _processor = std::make_shared<dataset::GenericBatchProcessor>(
        buildInputBlocks(/* for_single_inference= */ false), 
        buildLabelBlocks(/* for_single_inference= */ false, n_classes),
        /* has_header= */ false, /* delimiter= */ '\t');

    _inference_processor = std::make_shared<dataset::GenericBatchProcessor>(
        buildInputBlocks(/* for_single_inference= */ true), 
        buildLabelBlocks(/* for_single_inference= */ true),
        /* has_header= */ false, /* delimiter= */ '\t');
  }

  static std::vector<dataset::BlockPtr> buildInputBlocks(bool for_single_inference) {
    auto pairgram_encoding =
        std::make_shared<dataset::PairGram>(/* dim= */ 100000);
    uint32_t column = for_single_inference ? 0 : 1;
    return {std::make_shared<dataset::TextBlock>(
        column, pairgram_encoding)};
  }

  static std::vector<dataset::BlockPtr> buildLabelBlocks(bool for_single_inference, uint32_t n_classes=0) {
    if (!for_single_inference && n_classes == 0) {
      throw std::invalid_argument("buildLabelBlocks: Must pass n_classes if not for single inference.");
    }
    if (for_single_inference) {
      return {};
    }
    auto multi_label_encoding =
        std::make_shared<dataset::CategoricalMultiLabel>(
            /* n_classes= */ n_classes, /* delimiter= */ ',');
    return {std::make_shared<dataset::CategoricalBlock>(
        /* col= */ 0, /* encoding= */ multi_label_encoding)};
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
  std::shared_ptr<dataset::GenericBatchProcessor> _processor;
  std::shared_ptr<dataset::GenericBatchProcessor> _inference_processor;
  BoltGraphPtr _classifier;
};

}  // namespace thirdai::bolt