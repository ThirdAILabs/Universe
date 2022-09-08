#pragma once

#include "SequentialUtils.h"
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace thirdai::bolt::sequential_classifier {

class SequentialClassifier {
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
      const CategoricalPair& user, const CategoricalPair& target,
      const std::string& timestamp,
      const std::vector<std::string>& static_text = {},
      const std::vector<CategoricalPair>& static_categorical = {},
      const std::vector<SequentialTriplet>& sequential = {}) {
    _schema.user = user;
    _schema.target = target;
    _schema.timestamp_col_name = timestamp;
    _schema.static_text_col_names = static_text;
    _schema.static_categorical = static_categorical;
    _schema.sequential = sequential;

    _single_inference_col_nums = ColumnNumberMap(_schema);
    _single_inference_batch_processor =
        Pipeline::buildSingleInferenceBatchProcessor(
            _schema, _state, _single_inference_col_nums);
  }

  void train(const std::string& train_filename, uint32_t epochs,
             float learning_rate,
             std::vector<std::string> metrics = {"recall@1"}) {
    auto pipeline = Pipeline::buildForFile(_schema, _state, train_filename,
                                           /* delimiter = */ ',',
                                           /* for_training = */ true);

    auto output_sparsity = getLayerSparsity(pipeline.getLabelDim());

    if (!_model) {
      _model = CommonNetworks::FullyConnected(
          pipeline.getInputDim(),
          {FullyConnectedNode::make(/* dim= */ 512, /* activation= */ "relu"),
           FullyConnectedNode::make(pipeline.getLabelDim(), output_sparsity,
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

    _model->train({train_data}, train_labels, train_config);
  }

  InferenceResult predict(
      const std::string& test_filename,
      std::vector<std::string> metrics = {"recall@1"},
      const std::optional<std::string>& output_filename = std::nullopt) {
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
      uint32_t class_id = output.getHighestActivationId();
      auto target_lookup = _state.vocabs_by_column[_schema.target.first];
      (*output_file) << target_lookup->getString(class_id) << std::endl;
    };

    auto [test_data, test_labels] = pipeline.loadInMemory();

    PredictConfig config = PredictConfig::makeConfig()
                               .withMetrics(std::move(metrics))
                               .withOutputCallback(print_predictions_callback);

    auto results = _model->predict({test_data}, test_labels, config);

    if (output_file) {
      output_file->close();
    }

    return results;
  }

  std::tuple<std::vector<std::string>, std::vector<float>,
             std::vector<std::string>>
  explain(const std::unordered_map<std::string, std::string>& sample) {
    BoltVector input_vector = getInputForSingleInference(sample);

    auto [gradients_indices, gradients_ratios] =
        _model->getInputGradientSingle({input_vector});

    auto result = getPercentExplanationWithColumnNames(
        gradients_ratios, *gradients_indices,
        _single_inference_col_nums.getColumnNumToColNameMap(),
        _single_inference_batch_processor);

    return result;
  }

  BoltVector predictSingle(
      const std::unordered_map<std::string, std::string>& sample) {
    BoltVector input_vector = getInputForSingleInference(sample);

    return _model->predictSingle({input_vector},
                                 /* use_sparse_inference= */ false);
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
  BoltVector getInputForSingleInference(
      const std::unordered_map<std::string, std::string>& sample) {
    std::vector<std::string_view> columnar_sample(
        _single_inference_col_nums.size());
    for (const auto& [col_name, col_value] : sample) {
      uint32_t col_num = _single_inference_col_nums.at(col_name);
      columnar_sample[col_num] = col_value.data();
    }

    BoltVector input_vector;
    _single_inference_batch_processor->makeInputVector(columnar_sample,
                                                       input_vector);

    return input_vector;
  }
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

  Schema _schema;
  DataState _state;
  BoltGraphPtr _model;

  ColumnNumberMap _single_inference_col_nums;
  dataset::GenericBatchProcessorPtr _single_inference_batch_processor;

  // Private constructor for cereal
  SequentialClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_schema, _state, _model);
  }
};

}  // namespace thirdai::bolt::sequential_classifier

namespace thirdai::bolt {

using SequentialClassifier = sequential_classifier::SequentialClassifier;

}  // namespace thirdai::bolt