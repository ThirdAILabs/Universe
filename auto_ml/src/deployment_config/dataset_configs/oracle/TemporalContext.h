#pragma once

#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <_types/_uint32_t.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
namespace thirdai::automl::deployment {

class TemporalContext {
 public:
  TemporalContext() : _is_none(false) {}

  static TemporalContext None() { return TemporalContext(/* is_none= */ true); }

  bool isNone() const { return _is_none; }

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

  void initializeProcessor(dataset::GenericBatchProcessorPtr processor) {
    _processor = std::move(processor);
  }

  void update(const std::string& update) {
    if (!_processor) {
      throw std::invalid_argument(
          "Attempted to manually update temporal context before training.");
    }
    BoltVector vector;
    auto sample = dataset::ProcessorUtils::parseCsvRow(update, ',');
    // The following line updates the temporal context as a side effect,
    _processor->makeInputVector(sample, vector);
  }

  void batchUpdate(const std::vector<std::string>& updates) {
    if (!_processor) {
      throw std::invalid_argument(
          "Attempted to manually update temporal context before training.");
    }
    // The following line updates the temporal context as a side effect,
    _processor->createBatch(updates);
  }

 private:
  // Private constructor for none type.
  explicit TemporalContext(bool is_none) : _is_none(is_none) {}

  bool _is_none;
  std::unordered_map<uint32_t, dataset::QuantityHistoryTrackerPtr>
      _numerical_histories;
  std::unordered_map<uint32_t, dataset::ItemHistoryCollectionPtr>
      _categorical_histories;

  dataset::GenericBatchProcessorPtr _processor;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_numerical_histories, _categorical_histories, _processor);
  }
};

using TemporalContextPtr = std::shared_ptr<TemporalContext>;

}  // namespace thirdai::automl::deployment