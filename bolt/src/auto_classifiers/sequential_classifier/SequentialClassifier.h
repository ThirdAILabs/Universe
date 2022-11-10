#pragma once

#include "ConstructorUtilityTypes.h"
#include "SequentialClassifierConfig.h"
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
#include <utils/StringManipulation.h>
#include <chrono>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt::sequential_classifier {

class SequentialClassifierTextFixture;

// TODO(Geordie): Rename to UserPreferenceClassifier? PersonalizedRecommender?
// PersonalizedAndContextualizedClassifier?
class SequentialClassifier {
  friend SequentialClassifierTextFixture;

 public:
  SequentialClassifier(
      std::map<std::string, DataType> data_types,
      std::map<std::string,
               std::vector<std::variant<std::string, TemporalConfig>>>
          temporal_tracking_relationships,
      std::string target, std::string time_granularity = "d",
      uint32_t lookahead = 0) {
    _config.data_types = std::move(data_types),
    _config.target = std::move(target);
    _config.time_granularity =
        dataset::stringToGranularity(std::move(time_granularity));
    _config.lookahead = lookahead;
    if (!temporal_tracking_relationships.empty()) {
      autotuneTemporalFeatures(_config,
                               std::move(temporal_tracking_relationships));
    }
    _single_inference_col_nums = ColumnNumberMap(_config.data_types);
  }

  MetricData train(const std::string& train_filename, uint32_t epochs,
                   float learning_rate,
                   std::vector<std::string> metrics = {"recall@1"}) {
    auto pipeline =
        DataProcessing::buildDataLoaderForFile(_config, _state, train_filename,
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
          CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss(),
          /* print_when_done= */ false);
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

    auto pipeline =
        DataProcessing::buildDataLoaderForFile(_config, _state, test_filename,
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
      auto target_lookup = _state.vocabs_by_column[_config.target];

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

    EvalConfig config = EvalConfig::makeConfig()
                            .withMetrics(std::move(metrics))
                            .withOutputCallback(print_predictions_callback);

    auto results = _model->evaluate({test_data}, test_labels, config);

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
      std::optional<std::string> target_label = std::nullopt) {
    auto input_row = inputMapToInputRow(sample);

    auto processor = DataProcessing::buildSingleSampleBatchProcessor(
        _config, _state, _single_inference_col_nums,
        /* should_update_history= */ false);

    std::optional<uint32_t> neuron_to_explain;
    if (target_label) {
      auto label_vocab = _state.vocabs_by_column[_config.target];
      if (!label_vocab) {
        throw std::invalid_argument(
            "[Oracle::explain] called before training.");
      }
      neuron_to_explain = label_vocab->getUid(*target_label);
    }

    auto [gradients_indices, gradients_ratio] = _model->getInputGradientSingle(
        {makeInputForSingleInference(processor, input_row)}, true,
        neuron_to_explain);

    auto result = getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, input_row, processor);

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
          "[Oracle::predictSingle] k must be greater than or "
          "equal to 1.");
    }

    auto input_row = inputMapToInputRow(sample);

    auto processor = DataProcessing::buildSingleSampleBatchProcessor(
        _config, _state, _single_inference_col_nums,
        /* should_update_history= */ false);

    auto output = _model->predictSingle(
        {makeInputForSingleInference(processor, input_row)},
        /* use_sparse_inference= */ false);

    return outputVectorToTopKResults(output, k);
  }

  /**
   * @brief Indexes a single true sample to keep the SequentialClassifier's
   * internal quantity and category trackers up to date.
   *
   * @param sample A map from strings to strings, where the keys are column
   * names as specified in the SequentialClassifier schema and the values are
   * the values of the respective columns.
   */
  void indexSingle(const std::unordered_map<std::string, std::string>& sample) {
    auto input_row = inputMapToInputRow(sample);

    auto processor = DataProcessing::buildSingleSampleBatchProcessor(
        _config, _state, _single_inference_col_nums,
        /* should_update_history= */ true);

    // Emulate batch size of 2048.
    // TODO(Geordie): This is leaky abstraction.
    if (_state.n_index_single % 2048 == 0) {
      processor->prepareInputBlocksForBatch(input_row);
      _state.n_index_single = 0;
    }
    _state.n_index_single++;

    makeInputForSingleInference(processor, input_row);
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
    for (auto& str_view : input_row) {
      str_view = std::string_view(EMPTY_STR);
    }
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
      // Returns minimum element, so the results vector is going to
      // be sorted in ascending order.
      auto [activation, id] = top_k_activations.top();
      result.push_back(
          {_state.vocabs_by_column[_config.target]->getString(id), activation});
      top_k_activations.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
  }

  SequentialClassifierConfig _config;
  DataState _state;
  BoltGraphPtr _model;

  ColumnNumberMap _single_inference_col_nums;

  // Private constructor for cereal
  SequentialClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_config, _state, _model, _single_inference_col_nums);
  }

  static constexpr const char* EMPTY_STR = "";
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