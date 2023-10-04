#pragma once

#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <auto_ml/src/Aliases.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::automl {

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

  dataset::ItemHistoryCollectionPtr categoricalHistoryForId(uint32_t id) {
    if (!_categorical_histories.count(id)) {
      _categorical_histories[id] = dataset::ItemHistoryCollection::make();
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

  bool empty() const {
    return _numerical_histories.empty() && _categorical_histories.empty();
  }

 private:
  std::unordered_map<uint32_t, dataset::QuantityHistoryTrackerPtr>
      _numerical_histories;
  std::unordered_map<uint32_t, dataset::ItemHistoryCollectionPtr>
      _categorical_histories;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_numerical_histories, _categorical_histories);
  }
};

using TemporalContextPtr = std::shared_ptr<TemporalContext>;

}  // namespace thirdai::automl