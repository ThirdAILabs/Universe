#include "TemporalContext.h"
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleDatasetFactory.h>

namespace thirdai::automl::deployment {

dataset::QuantityHistoryTrackerPtr TemporalContext::numericalHistoryForId(
    uint32_t id, uint32_t lookahead, uint32_t history_length,
    dataset::QuantityTrackingGranularity time_granularity) {
  if (!_numerical_histories.count(id)) {
    _numerical_histories[id] = dataset::QuantityHistoryTracker::make(
        lookahead, history_length, time_granularity);
  }
  return _numerical_histories[id];
}

dataset::ItemHistoryCollectionPtr TemporalContext::categoricalHistoryForId(
    uint32_t id, uint32_t n_users) {
  if (!_categorical_histories.count(id)) {
    _categorical_histories[id] = dataset::ItemHistoryCollection::make(n_users);
  }
  return _categorical_histories[id];
}

void TemporalContext::reset() {
  for (auto& [_, history] : _numerical_histories) {
    history->reset();
  }
  for (auto& [_, history] : _categorical_histories) {
    history->reset();
  }
}

void TemporalContext::updateTemporalTrackers(const std::string& update) {
  _featurizer->featurizeInput(update, /* should_update_history= */ true);
}

void TemporalContext::updateTemporalTrackers(const MapInput& update) {
  _featurizer->featurizeInput(update, /* should_update_history= */ true);
}

void TemporalContext::batchUpdateTemporalTrackers(
    const std::vector<std::string>& updates) {
  _featurizer->featurizeInputBatch(updates, /* should_update_history= */ true);
}

void TemporalContext::batchUpdateTemporalTrackers(
    const MapInputBatch& updates) {
  _featurizer->featurizeInputBatch(updates, /* should_update_history= */ true);
}

using TemporalContextPtr = std::shared_ptr<TemporalContext>;

}  // namespace thirdai::automl::deployment