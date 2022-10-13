#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Datasets.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

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

class SaveContext {
 public:
  SaveContext(std::string prefix, uint32_t frequency)
      : _prefix(std::move(prefix)), _frequency(frequency) {}
  const std::string& prefix() const { return _prefix; }
  uint32_t frequency() const { return _frequency; }

 private:
  std::string _prefix;
  uint32_t _frequency;
};

class ValidationContext {
 public:
  explicit ValidationContext(
      std::vector<dataset::BoltDatasetPtr> validation_data,
      dataset::BoltDatasetPtr validation_labels, PredictConfig predict_config,
      uint32_t frequency)
      : _data(std::move(validation_data)),
        _labels(std::move(validation_labels)),
        _config(std::move(predict_config)),
        _frequency(frequency) {}

  const std::vector<dataset::BoltDatasetPtr>& data() const { return _data; }

  const dataset::BoltDatasetPtr& labels() const { return _labels; }

  const PredictConfig& config() const { return _config; }

  uint32_t frequency() const { return _frequency; }

 private:
  std::vector<dataset::BoltDatasetPtr> _data;
  dataset::BoltDatasetPtr _labels;
  PredictConfig _config;
  uint32_t _frequency;
};

class TrainConfig;
using TrainConfigPtr = std::shared_ptr<TrainConfig>;

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
      const PredictConfig& predict_config, uint32_t validation_frequency = 0) {
    _validation_context =
        ValidationContext(validation_data, validation_labels, predict_config,
                          validation_frequency);
    return *this;
  }

  TrainConfig& withLogLossFrequency(uint32_t log_loss_frequency) {
    _log_loss_frequency = log_loss_frequency;
    return *this;
  }

  TrainConfig& withSaveParameters(const std::string& save_prefix,
                                  uint32_t save_frequency) {
    _save_context = SaveContext(save_prefix, save_frequency);
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

  void save(const std::string& filename) const {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    save_stream(filestream);
  }

  void save_stream(std::ostream& output_stream) const {
    if (_callbacks.numCallbacks() != 0) {
      throw std::runtime_error(
          "Cannot serialize a training config that has callbacks.");
    }
    if (_validation_context.has_value()) {
      throw std::runtime_error(
          "Cannot serialize a training config that has a validation context.");
    }
    cereal::BinaryOutputArchive oarchive(output_stream);
    oarchive(*this);
  }

  static TrainConfigPtr load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    return load_stream(filestream);
  }

  static TrainConfigPtr load_stream(std::istream& input_stream) {
    cereal::BinaryInputArchive iarchive(input_stream);
    std::shared_ptr<TrainConfig> deserialize_into(new TrainConfig());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

  uint32_t logLossFrequency() const { return _log_loss_frequency; }

  void setEpochs(uint32_t new_epochs) { _epochs = new_epochs; }

  const std::optional<SaveContext>& saveContext() const {
    return _save_context;
  }

 private:
  // Private constructor for cereal.
  TrainConfig() : TrainConfig(0, 0){};

  TrainConfig(float learning_rate, uint32_t epochs)
      : _epochs(epochs),
        _learning_rate(learning_rate),
        _metric_names({}),
        _verbose(true),
        _rebuild_hash_tables(std::nullopt),
        _reconstruct_hash_functions(std::nullopt),
        _callbacks({}),
        _validation_context(std::nullopt),
        _save_context(std::nullopt),
        _log_loss_frequency(1) {}

  friend class cereal::access;
  // We don't serialize the callbacks because they might be arbitrary functions
  // from python. Instead, we throw an error in the save method if the callbacks
  // are nonempty.
  // For now, we also don't serialize the validation context, and similarly
  // check this in the save method.
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_epochs, _learning_rate, _metric_names, _verbose,
            _rebuild_hash_tables, _reconstruct_hash_functions);
  }

  uint32_t _epochs;
  float _learning_rate;
  std::vector<std::string> _metric_names;
  bool _verbose;

  std::optional<uint32_t> _rebuild_hash_tables;
  std::optional<uint32_t> _reconstruct_hash_functions;

  CallbackList _callbacks;

  std::optional<ValidationContext> _validation_context;
  std::optional<SaveContext> _save_context;

  /// Log loss frequency, in units of updates (1 batch = 1 update).
  uint32_t _log_loss_frequency;
};

class TrainState {
 public:
  TrainState(const TrainConfig& train_config, uint32_t batch_size,
             uint32_t data_len)
      : learning_rate(train_config.learningRate()),
        epoch(train_config.epochs()),
        batch_cnt(0),
        verbose(train_config.verbose()),
        rebuild_hash_tables_batch(
            train_config.getRebuildHashTablesBatchInterval(batch_size,
                                                           data_len)),
        reconstruct_hash_functions_batch(
            train_config.getReconstructHashFunctionsBatchInterval(batch_size,
                                                                  data_len)),
        stop_training(false),
        train_metric_aggregator(train_config.getMetricAggregator()) {}

  float learning_rate;
  uint32_t epoch;
  uint64_t batch_cnt;
  bool verbose;

  uint32_t rebuild_hash_tables_batch;
  uint32_t reconstruct_hash_functions_batch;

  bool stop_training;

  std::vector<double> epoch_times;

  MetricAggregator& getTrainMetricAggregator() {
    return train_metric_aggregator;
  }

  void updateValidationMetrics(const InferenceMetricData& metric_data) {
    for (const auto& [metric_name, value] : metric_data) {
      validation_metrics[metric_name].push_back(value);
    }
  }

  const std::vector<double>& getTrainMetrics(const std::string& metric_name) {
    return train_metric_aggregator.getSingleOutput(metric_name);
  }

  MetricData getAllTrainMetrics() {
    return train_metric_aggregator.getOutput();
  }

  const std::vector<double>& getValidationMetrics(
      const std::string& metric_name) {
    if (validation_metrics.count(metric_name) != 0) {
      return validation_metrics[metric_name];
    }
    throw std::invalid_argument("Could not find metric name '" + metric_name +
                                "' in list of computed validation metrics. ");
  }

  const auto& getAllValidationMetrics() { return validation_metrics; }

 private:
  MetricAggregator train_metric_aggregator;
  std::unordered_map<std::string, std::vector<double>> validation_metrics;
};

}  // namespace thirdai::bolt
