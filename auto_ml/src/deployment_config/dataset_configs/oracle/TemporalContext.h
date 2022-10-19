#pragma once

#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/Aliases.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/Conversions.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace thirdai::automl::deployment {

class TemporalContext {
 public:
  TemporalContext() {}

  dataset::QuantityHistoryTrackerPtr numericalHistoryForId(
      uint32_t id, uint32_t lookahead, uint32_t history_length,
      dataset::QuantityTrackingGranularity time_granularity) {
    if (!_numerical_histories.count(id)) {
      _numerical_histories[id] = dataset::QuantityHistoryTracker::make(
          lookahead, history_length, time_granularity);
    }
    return _numerical_histories[id];
  }

  dataset::ItemHistoryCollectionPtr categoricalHistoryForId(uint32_t id,
                                                            uint32_t n_users) {
    if (!_categorical_histories.count(id)) {
      _categorical_histories[id] =
          dataset::ItemHistoryCollection::make(n_users);
    }
    return _categorical_histories[id];
  }

  void reset() {
    for (auto& [_, history] : _numerical_histories) {
      history->reset();
    }
    for (auto& [_, history] : _categorical_histories) {
      history->reset();
    }
  }

  void initializeDataStructures(dataset::GenericBatchProcessorPtr processor,
                                ColumnNumberMapPtr column_number_map,
                                char delimiter) {
    if (!_processor) {
      _processor = std::move(processor);
      _column_number_map = std::move(column_number_map);
      _delimiter = delimiter;
    } else if (_processor != processor ||
               _column_number_map != column_number_map ||
               _delimiter != delimiter) {
      throw std::invalid_argument(
          "Temporal context already initialized with different data "
          "structures.");
    }
  }

  void updateTemporalTrackers(const std::string& update) {
    if (!_processor) {
      throw std::invalid_argument(
          "Attempted to manually update temporal context before training.");
    }

    auto sample =
        ConversionUtils::stringInputToVectorOfStringViews(update, _delimiter);

    BoltVector vector;
    // The following line updates the temporal context as a side effect,
    if (auto exception = _processor->makeInputVector(sample, vector)) {
      std::rethrow_exception(exception);
    }
  }

  void updateTemporalTrackers(const MapInput& update) {
    if (!_processor) {
      throw std::invalid_argument(
          "Attempted to manually update temporal context before training.");
    }

    auto sample = ConversionUtils::mapInputToVectorOfStringViews(
        update, *_column_number_map);

    BoltVector vector;
    // The following line updates the temporal context as a side effect,
    if (auto exception = _processor->makeInputVector(sample, vector)) {
      std::rethrow_exception(exception);
    }
  }

  void batchUpdateTemporalTrackers(const std::vector<std::string>& updates) {
    if (!_processor) {
      throw std::invalid_argument(
          "Attempted to manually update temporal context before training.");
    }
    // The following line updates the temporal context as a side effect,
    _processor->createBatch(updates);
  }

  void batchUpdateTemporalTrackers(const MapInputBatch& updates) {
    if (!_processor) {
      throw std::invalid_argument(
          "Attempted to manually update temporal context before training.");
    }
    auto string_batch = ConversionUtils::mapInputBatchToStringBatch(
        updates, _delimiter, *_column_number_map);
    // The following line updates the temporal context as a side effect,
    _processor->createBatch(string_batch);
  }

 private:
  static inline std::string EMPTY;

  std::unordered_map<uint32_t, dataset::QuantityHistoryTrackerPtr>
      _numerical_histories;
  std::unordered_map<uint32_t, dataset::ItemHistoryCollectionPtr>
      _categorical_histories;

  dataset::GenericBatchProcessorPtr _processor;
  ColumnNumberMapPtr _column_number_map;
  char _delimiter;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_numerical_histories, _categorical_histories, _processor,
            _column_number_map, _delimiter);
  }
};

using TemporalContextPtr = std::shared_ptr<TemporalContext>;

}  // namespace thirdai::automl::deployment