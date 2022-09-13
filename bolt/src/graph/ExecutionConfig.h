#pragma once

#include <bolt/src/graph/callbacks/Callback.h>
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
    return MetricAggregator(_metric_names);
  }

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

class ValidationContext {
 public:
  explicit ValidationContext(
      const std::vector<dataset::BoltDatasetPtr>& _validation_data,
      const dataset::BoltDatasetPtr& _validation_labels,
      const PredictConfig& _predict_config)
      : _data(_validation_data),
        _labels(_validation_labels),
        _config(_predict_config) {}

  const std::vector<dataset::BoltDatasetPtr>& data() const { return _data; }

  const dataset::BoltDatasetPtr& labels() const { return _labels; }

  const PredictConfig& config() const { return _config; }

 private:
  std::vector<dataset::BoltDatasetPtr> _data;
  dataset::BoltDatasetPtr _labels;
  PredictConfig _config;
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

  TrainConfig& withCallbacks(const std::vector<CallbackPtr>& callbacks) {
    _callbacks = CallbackList(callbacks);
    return *this;
  }

  TrainConfig& withValidation(
      const std::vector<dataset::BoltDatasetPtr>& validation_data,
      const dataset::BoltDatasetPtr& validation_labels,
      const PredictConfig& predict_config) {
    _validation_context =
        ValidationContext(validation_data, validation_labels, predict_config);
    return *this;
  }

  std::optional<ValidationContext> getValidationContext() const {
    return _validation_context;
  }

  CallbackList getCallbacks() const { return _callbacks; }

  constexpr uint32_t epochs() const { return _epochs; }

  constexpr float learningRate() const { return _learning_rate; }

  MetricAggregator getMetricAggregator() const {
    return MetricAggregator(_metric_names);
  }

  constexpr bool verbose() const { return _verbose; }

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
        _rebuild_hash_tables(std::nullopt),
        _reconstruct_hash_functions(std::nullopt),
        _callbacks({}),
        _validation_context(std::nullopt) {}

  uint32_t _epochs;
  float _learning_rate;
  std::vector<std::string> _metric_names;
  bool _verbose;

  std::optional<uint32_t> _rebuild_hash_tables;
  std::optional<uint32_t> _reconstruct_hash_functions;

  CallbackList _callbacks;

  std::optional<ValidationContext> _validation_context;
};

class TrainState {
 public:
  TrainState(const TrainConfig& train_config, uint32_t batch_size,
             uint32_t data_len)
      : learning_rate(train_config.learningRate()),
        verbose(train_config.verbose()),
        rebuild_hash_tables_batch(
            train_config.getRebuildHashTablesBatchInterval(batch_size,
                                                           data_len)),
        reconstruct_hash_functions_batch(
            train_config.getReconstructHashFunctionsBatchInterval(batch_size,
                                                                  data_len)),
        stop_training(false) {}

  float learning_rate;
  bool verbose;

  uint32_t rebuild_hash_tables_batch;
  uint32_t reconstruct_hash_functions_batch;

  bool stop_training;

  void updateTrainMetrics(const MetricData& metric_data) {
    for (const auto& [metric_name, value] : metric_data) {
      metrics["train_" + metric_name] = value;
    }
  }

  void updateValidationMetrics(const InferenceMetricData& metric_data) {
    for (const auto& [metric_name, value] : metric_data) {
      metrics["val_" + metric_name].push_back(value);
    }
  }

  std::vector<double> getMetricValues(const std::string& metric_name) {
    if (metrics.count(metric_name) == 0) {
      throw std::invalid_argument(
          "Could not find metric name '" + metric_name +
          "' in list of computed metrics. Metric names are the same as those "
          "passed in but prefixed with 'train_' and 'val_' depending on the "
          "association to training/validation respectively. ");
    }
    return metrics[metric_name];
  }

  void updateEpochTimes(int64_t epoch_time) {
    epoch_times.push_back(static_cast<double>(epoch_time));
  }

  std::vector<double> getEpochTimes() { return epoch_times; }

 private:
  std::unordered_map<std::string, std::vector<double>> metrics;
  std::vector<double> epoch_times;
};

}  // namespace thirdai::bolt
