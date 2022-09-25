#pragma once

#include "SequentialUtils.h"
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt::sequential_classifier {

class SequentialClassifierTextFixture;

class SequentialClassifier {
  friend SequentialClassifierTextFixture;

 public:
  /**
   * @brief Construct a new Sequential Classifier object
   *
   * @param user A (string, unsigned integer) pair cantaining
   * the user column name and the number of unique users
   * respectively.
   * @param target A (string, unsigned integer) pair cantaining
   * the target column name and the number of unique target
   * classes respectively.
   * @param timestamp The name of the column that contains
   * timestamps
   * @param static_text A vector of names of columns that
   * contain static textual information.
   * @param static_categorical A vector of
   * (string, unsigned integer) pairs containing static
   * categorical column name and the number of unique classes
   * respectively.
   * @param sequential A vector of
   * (string, unsigned integer, unsigned integer) tuples containing
   * sequential column name, the number of unique classes, and
   * the number of previous values to track.
   */
  SequentialClassifier(
      CategoricalPair user, CategoricalPair target, std::string timestamp,
      std::vector<std::string> static_text = {},
      std::vector<CategoricalPair> static_categorical = {},
      std::vector<SequentialTriplet> sequential = {},
      std::vector<std::string> dense_sequential = {},
      std::optional<char> multi_class_delim = std::nullopt,
      std::optional<uint32_t> history_lag = std::nullopt,
      std::optional<uint32_t> history_length = std::nullopt,
      dataset::QuantityTrackingGranularity tracking_granularity = dataset::QuantityTrackingGranularity::Daily) {
    _schema.user = std::move(user);
    _schema.target = std::move(target);
    _schema.timestamp_col_name = std::move(timestamp);
    _schema.static_text_col_names = std::move(static_text);
    _schema.static_categorical = std::move(static_categorical);
    _schema.sequential = std::move(sequential);
    _schema.dense_sequential = std::move(dense_sequential);
    _schema.multi_class_delim = multi_class_delim;
    _schema.history_lag = history_lag;
    _schema.history_length = history_length;
    _schema.history_granularity = tracking_granularity;

    _single_inference_col_nums = ColumnNumberMap(_schema);
  }

  MetricData train(const std::string& train_filename, uint32_t epochs,
                   float learning_rate,
                   std::vector<std::string> metrics = {"recall@1"}) {
    auto pipeline = Pipeline::buildForFile(_schema, _state, train_filename,
                                           /* delimiter = */ ',',
                                           /* for_training = */ true);

    auto output_sparsity = getLayerSparsity(pipeline.getLabelDim());

    if (!_model) {
      _model = CommonNetworks::FullyConnected(
          pipeline.getInputDim(),
          {FullyConnectedNode::makeDense(/* dim= */ 512,
                                         /* activation= */ "relu"),
           FullyConnectedNode::makeExplicitSamplingConfig(
               pipeline.getLabelDim(), output_sparsity,
               /* activation= */ "softmax",
               /* num_tables= */ 64,
               /* hashes_per_table= */ 4,
               /* reservoir_size= */ 64)});
      _model->compile(
          CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());
    }

    auto [train_data, train_labels] = pipeline.loadInMemory();

    TrainConfig train_config =
        TrainConfig::makeConfig(/* learning_rate= */ learning_rate,
                                /* epochs= */ epochs)
            .withMetrics(std::move(metrics));

    return _model->train({train_data}, train_labels, train_config);
  }

  InferenceMetricData predict(
      const std::string& test_filename,
      std::vector<std::string> metrics = {"recall@1"},
      const std::optional<std::string>& output_filename = std::nullopt,
      uint32_t print_top_k = 1) {
    if (!_model) {
      throw std::runtime_error("Called predict() before training.");
    }

    auto pipeline = Pipeline::buildForFile(_schema, _state, test_filename,
                                           /* delimiter = */ ',',
                                           /* for_training = */ false);

    std::optional<std::ofstream> output_file;
    if (output_filename) {
      output_file = dataset::SafeFileIO::ofstream(*output_filename);
    }

    auto print_predictions_callback = [&](const BoltVector& output) {
      if (!output_file) {
        return;
      }
      auto class_ids = output.findKLargestActivations(print_top_k);
      auto target_lookup = _state.vocabs_by_column[_schema.target.first];

      uint32_t first = true;
      while (!class_ids.empty()) {
        auto [_activation, class_id] = class_ids.top();
        class_ids.pop();
        if (!first) {
          (*output_file) << ',';
        }
        (*output_file) << target_lookup->getString(class_id) << std::endl;
        first = false;
      }
    };

    auto [test_data, test_labels] = pipeline.loadInMemory();

    PredictConfig config = PredictConfig::makeConfig()
                               .withMetrics(std::move(metrics))
                               .withOutputCallback(print_predictions_callback);

    auto results = _model->predict({test_data}, test_labels, config);

    if (output_file) {
      output_file->close();
    }

    return results.first;
  }

  std::string summarizeModel() {
    if (!_model) {
      throw std::runtime_error("Called sumarizeModel() before training.");
    }

    return _model->summarize(/* print= */ false, /* detailed= */ true);
  }

  std::vector<dataset::Explanation> explain(
      const std::unordered_map<std::string, std::string>& sample,
      std::optional<uint32_t> neuron_to_explain = std::nullopt) {
    auto input_row = inputMapToInputRow(sample);

    auto processor = Pipeline::buildSingleInferenceBatchProcessor(
        _schema, _state, _single_inference_col_nums);

    auto result = getSignificanceSortedExplanations(
        _model, makeInputForSingleInference(processor, input_row), input_row,
        processor, neuron_to_explain);

    auto column_num_to_name =
        _single_inference_col_nums.getColumnNumToColNameMap();

    for (auto& response : result) {
      response.column_name = column_num_to_name[response.column_number];
    }

    return result;
  }

  /**
   * @brief Computes the top k classes and their probabilities.
   *
   * @param sample A map from strings to strings, where the keys are column
   * names as specified in the SequentialClassifier schema and the values are
   * the values of the respective columns.
   * @param k The number of top results to return.
   * @return std::vector<std::pair<std::string, float>> A vector of
   * (class name. probability) pairs.
   */
  std::vector<std::pair<std::string, float>> predictSingle(
      const std::unordered_map<std::string, std::string>& sample,
      uint32_t k = 1) {
    if (k < 1) {
      throw std::invalid_argument(
          "[SequentialClassifier::predictSingle] k must be greater than or "
          "equal to 1.");
    }

    auto input_row = inputMapToInputRow(sample);

    auto processor = Pipeline::buildSingleInferenceBatchProcessor(
        _schema, _state, _single_inference_col_nums);

    auto output = _model->predictSingle(
        {makeInputForSingleInference(processor, input_row)},
        /* use_sparse_inference= */ false);

    return outputVectorToTopKResults(output, k);
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<SequentialClassifier> load(
      const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<SequentialClassifier> deserialize_into(
        new SequentialClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
  static float getLayerSparsity(uint32_t layer_size) {
    if (layer_size < 500) {
      return 1.0;
    }
    if (layer_size < 1000) {
      return 0.2;
    }
    if (layer_size < 2000) {
      return 0.1;
    }
    if (layer_size < 5000) {
      return 0.05;
    }
    if (layer_size < 10000) {
      return 0.02;
    }
    if (layer_size < 20000) {
      return 0.01;
    }
    return 0.005;
  }

  std::vector<std::string_view> inputMapToInputRow(
      const std::unordered_map<std::string, std::string>& input_map) {
    std::vector<std::string_view> input_row(_single_inference_col_nums.size());
    for (const auto& [col_name, col_value] : input_map) {
      uint32_t col_num = _single_inference_col_nums.at(col_name);
      input_row[col_num] = col_value.data();
    }
    return input_row;
  }

  static BoltVector makeInputForSingleInference(
      const dataset::GenericBatchProcessorPtr& processor,
      std::vector<std::string_view>& input_row) {
    BoltVector input_vector;
    processor->makeInputVector(input_row, input_vector);
    return input_vector;
  }

  std::vector<std::pair<std::string, float>> outputVectorToTopKResults(
      const BoltVector& output, uint32_t k) {
    auto top_k_activations = output.findKLargestActivations(k);

    std::vector<std::pair<std::string, float>> result;
    result.reserve(k);

    while (!top_k_activations.empty()) {
      auto [activation, id] = top_k_activations.top();
      result.push_back(
          {_state.vocabs_by_column[_schema.target.first]->getString(id),
           activation});
      top_k_activations.pop();
    }
    return result;
  }

  Schema _schema;
  DataState _state;
  BoltGraphPtr _model;

  ColumnNumberMap _single_inference_col_nums;

  // Private constructor for cereal
  SequentialClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_schema, _state, _model, _single_inference_col_nums);
  }
};

class SequentialClassifierTextFixture {
 public:
  static DataState getState(const SequentialClassifier& model) {
    return model._state;
  }
};

}  // namespace thirdai::bolt::sequential_classifier

namespace thirdai::bolt {

using SequentialClassifier = sequential_classifier::SequentialClassifier;

}  // namespace thirdai::bolt