#pragma once

#include <bolt/src/metrics/MetricAggregator.h>
#include <limits>
#include <optional>

namespace thirdai::bolt {

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

  // TrainConfig& withEarlyStopValidation() {

  // }

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
        _reconstruct_hash_functions(std::nullopt) {}

  uint32_t _epochs;
  float _learning_rate;
  std::vector<std::string> _metric_names;
  bool _verbose;
  std::optional<uint32_t> _batch_size;

  std::optional<uint32_t> _rebuild_hash_tables;
  std::optional<uint32_t> _reconstruct_hash_functions;
};

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

}  // namespace thirdai::bolt