#pragma once

#include <bolt/src/auto_classifiers/AutoClassifierBase.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/FullyConnectedGraphNetwork.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
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
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

class WayfairClassifier {
 public:
  explicit WayfairClassifier(uint32_t n_classes) : _n_classes(n_classes) {
    buildBatchProcessors(n_classes);

    assert(n_classes == _processor->getLabelDim());

    std::vector<std::pair<uint32_t, float>> hidden_layer_config = {{1024, 1.0}};

    _classifier = FullyConnectedGraphNetwork::build(
        /* input_dim= */ _processor->getInputDim(),
        /* hidden_dims_and_sparsities= */ hidden_layer_config,
        /* output_dim= */ n_classes,
        /* output_sparsity= */ n_classes >= 500 ? 0.1 : 1,
        /* output_activation= */ "sigmoid",
        /* loss= */ std::make_shared<BinaryCrossEntropyLoss>());
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
        TrainConfig::makeConfig(learning_rate, epochs).withMetrics(metrics);

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
    auto sample = sentenceToSample(sentence);

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

  static std::unique_ptr<WayfairClassifier> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<WayfairClassifier> deserialize_into(
        new WayfairClassifier());
    iarchive(*deserialize_into);
    deserialize_into->buildBatchProcessors(deserialize_into->_n_classes);
    return deserialize_into;
  }

 protected:
  void buildBatchProcessors(uint32_t n_classes) {
    auto multi_label_encoding =
        std::make_shared<dataset::CategoricalMultiLabel>(
            /* n_classes= */ n_classes, /* delimiter= */ ',');
    auto label_block = std::make_shared<dataset::CategoricalBlock>(
        /* col= */ 0, /* encoding= */ multi_label_encoding);
    std::vector<std::shared_ptr<dataset::Block>> label_blocks = {label_block};

    auto pairgram_encoding =
        std::make_shared<dataset::PairGram>(/* dim= */ 100000);
    auto input_block = std::make_shared<dataset::TextBlock>(
        /* col= */ 1, /* encoding= */ pairgram_encoding);
    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {input_block};

    _processor = std::make_shared<dataset::GenericBatchProcessor>(
        input_blocks, label_blocks,
        /* has_header= */ false, /* delimiter= */ '\t');
  }

  static std::string tokensToSentence(const std::vector<uint32_t>& tokens) {
    std::stringstream sentence_ss;
    char delim = '\t';
    for (auto token : tokens) {
      sentence_ss << delim << token;
      delim = ' ';
    }
    return sentence_ss.str();
  }

  static std::vector<std::string_view> sentenceToSample(
      const std::string& sentence) {
    return {std::string_view(sentence.data(), 1),
            std::string_view(sentence.data() + 1, sentence.size() - 1)};
  }

  // Private constructor for cereal
  WayfairClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_n_classes, _classifier);
  }

  uint32_t _n_classes;
  std::shared_ptr<dataset::GenericBatchProcessor> _processor;
  BoltGraphPtr _classifier;
};

}  // namespace thirdai::bolt