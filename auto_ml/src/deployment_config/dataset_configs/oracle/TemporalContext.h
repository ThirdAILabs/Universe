#pragma once

#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace thirdai::automl::deployment {

class Featurizer;
using FeaturizerPtr = std::shared_ptr<Featurizer>;

class TemporalContext {
 public:
  explicit TemporalContext(FeaturizerPtr featurizer)
      : _featurizer(std::move(featurizer)) {}

  dataset::QuantityHistoryTrackerPtr numericalHistoryForId(
      uint32_t id, uint32_t lookahead, uint32_t history_length,
      dataset::QuantityTrackingGranularity time_granularity);

  dataset::ItemHistoryCollectionPtr categoricalHistoryForId(uint32_t id,
                                                            uint32_t n_users);

  void reset();

  void updateTemporalTrackers(const std::string& update);

  void batchUpdateTemporalTrackers(const std::vector<std::string>& updates);

 private:
  TemporalContext() {}

  std::unordered_map<uint32_t, dataset::QuantityHistoryTrackerPtr>
      _numerical_histories;
  std::unordered_map<uint32_t, dataset::ItemHistoryCollectionPtr>
      _categorical_histories;

  FeaturizerPtr _featurizer;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_numerical_histories, _categorical_histories, _featurizer);
  }
};

using TemporalContextPtr = std::shared_ptr<TemporalContext>;

}  // namespace thirdai::automl::deployment