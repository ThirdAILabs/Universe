#pragma once

#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Datasets.h>
#include <limits>
#include <optional>

namespace thirdai::bolt {

class PredictConfig {
 public:
  static PredictConfig makeConfig() { return PredictConfig(); }

  PredictConfig& enableSparseInference() {
    _use_sparse_inference = true;
    return *this;
  }

  PredictConfig& withMetrics(std::vector<std::string> metric_names) {
    _metric_names = std::move(metric_names);
    return *this;
  }

  PredictConfig& returnActivations() {
    _return_activations = true;
    return *this;
  }

  PredictConfig& silence() {
    _verbose = false;
    return *this;
  }

  PredictConfig& withOutputCallback(
      const std::function<void(const BoltVector&)>& output_callback) {
    _output_callback = output_callback;
    return *this;
  }

  bool sparseInferenceEnabled() const { return _use_sparse_inference; }

  MetricAggregator getMetricAggregator() const {
    return MetricAggregator(_metric_names, _verbose);
  }

  uint32_t getNumMetricsTracked() const { return _metric_names.size(); }

  std::vector<std::string> getMetricNames() const { return _metric_names; }

  constexpr bool verbose() const { return _verbose; }

  constexpr bool shouldReturnActivations() const { return _return_activations; }

  auto outputCallback() const { return _output_callback; }

 private:
  PredictConfig()
      : _metric_names({}),
        _use_sparse_inference(false),
        _verbose(true),
        _return_activations(false),
        _output_callback(std::nullopt) {}

  std::vector<std::string> _metric_names;
  bool _use_sparse_inference, _verbose, _return_activations;
  std::optional<std::function<void(const BoltVector&)>> _output_callback;
};

class TrainConfig {
 public:
  /*
    The parameters epochs and learning rate must be explicitly provided to
    construct the training config. The remaining parameters can be set using a
    builder pattern.
  */

  static TrainConfig makeConfig(float learning_rate, uint32_t epochs) {
    return TrainConfig(learning_rate, epochs);
  }

  TrainConfig& withMetrics(std::vector<std::string> metric_names) {
    _metric_names = std::move(metric_names);
    return *this;
  }

  TrainConfig& withBatchSize(uint32_t batch_size) {
    _batch_size = batch_size;
    return *this;
  }

  TrainConfig& silence() {
    _verbose = false;
    return *this;
  }

  TrainConfig& withRebuildHashTables(uint32_t rebuild) {
    _rebuild_hash_tables = rebuild;
    return *this;
  }

  TrainConfig& withReconstructHashFunctions(uint32_t reconstruct) {
    _reconstruct_hash_functions = reconstruct;
    return *this;
  }

  constexpr uint32_t epochs() const { return _epochs; }

  constexpr float learningRate() const { return _learning_rate; }

  MetricAggregator getMetricAggregator() const {
    return MetricAggregator(_metric_names, _verbose);
  }

  constexpr bool verbose() const { return _verbose; }

  struct EarlyStopValidationMetadata {
    inline static const std::string BEST_MODEL_SAVE_LOCATION =
        "bestModelSaveLocation";

    EarlyStopValidationMetadata(
        const std::vector<dataset::BoltDatasetPtr>&& valid_data,
        const std::vector<dataset::BoltTokenDatasetPtr>&& valid_tokens,
        const dataset::BoltDatasetPtr&& valid_labels, uint32_t patience,
        const PredictConfig& predict_config)
        : valid_data(valid_data),
          valid_tokens(valid_tokens),
          valid_labels(valid_labels),
          patience(patience),
          predict_config(predict_config),
          best_validation_metric(0),
          last_validation_metric(0) {}

    std::vector<dataset::BoltDatasetPtr> valid_data;
    std::vector<dataset::BoltTokenDatasetPtr> valid_tokens;
    dataset::BoltDatasetPtr valid_labels;
    uint32_t patience;
    PredictConfig predict_config;

    double best_validation_metric;
    double last_validation_metric;
  };

  TrainConfig& withEarlyStopValidation(
      const std::vector<dataset::BoltDatasetPtr>& valid_data,
      const std::vector<dataset::BoltTokenDatasetPtr>& valid_tokens,
      const dataset::BoltDatasetPtr& valid_labels,
      const PredictConfig& predict_config, uint32_t patience = 3) {
    uint32_t num_metrics = predict_config.getNumMetricsTracked();
    if (num_metrics != 1) {
      throw std::invalid_argument(
          "Validation-based early stopping only supports the use of one "
          "metric, passed in " +
          std::to_string(num_metrics) + " metrics.");
    }

    _early_stop_metadata = EarlyStopValidationMetadata(
        std::move(valid_data), std::move(valid_tokens), std::move(valid_labels),
        patience, predict_config);
    return *this;
  }

  auto getEarlyStopValidationMetadata() const { return _early_stop_metadata; }

  constexpr bool usingEarlyStopValidation() const {
    return _early_stop_metadata.has_value();
  }

  uint32_t getRebuildHashTablesBatchInterval(uint32_t batch_size,
                                             uint32_t data_len) const {
    constexpr uint32_t LargeDatasetThreshold = 100000;
    constexpr uint32_t LargeDatasetFactor = 100;
    constexpr uint32_t SmallDatasetFactor = 20;

    uint32_t rebuild_param;

    if (!_rebuild_hash_tables) {
      // For larger datasts we want to do more frequent hash table updates.
      if (data_len < LargeDatasetThreshold) {
        rebuild_param = data_len / SmallDatasetFactor;
      } else {
        rebuild_param = data_len / LargeDatasetFactor;
      }
    } else {
      rebuild_param = _rebuild_hash_tables.value();
    }

    return std::max<uint32_t>(rebuild_param / batch_size, 1);
  }

  uint32_t getReconstructHashFunctionsBatchInterval(uint32_t batch_size,
                                                    uint32_t data_len) const {
    // If reconstruct_hash_functions is not provided then we will have it
    // reconstruct the hash functions every time it process a quarter of the
    // dataset.
    uint32_t reconstruct_param =
        _reconstruct_hash_functions.value_or(data_len / 4);

    return std::max<uint32_t>(reconstruct_param / batch_size, 1);
  }

 private:
  TrainConfig(float learning_rate, uint32_t epochs)
      : _epochs(epochs),
        _learning_rate(learning_rate),
        _metric_names({}),
        _verbose(true),
        _batch_size({}),
        _rebuild_hash_tables(std::nullopt),
        _reconstruct_hash_functions(std::nullopt),
        _early_stop_metadata(std::nullopt) {}

  uint32_t _epochs;
  float _learning_rate;
  std::vector<std::string> _metric_names;
  bool _verbose;
  std::optional<uint32_t> _batch_size;

  std::optional<uint32_t> _rebuild_hash_tables;
  std::optional<uint32_t> _reconstruct_hash_functions;

  std::optional<EarlyStopValidationMetadata> _early_stop_metadata;
};

}  // namespace thirdai::bolt