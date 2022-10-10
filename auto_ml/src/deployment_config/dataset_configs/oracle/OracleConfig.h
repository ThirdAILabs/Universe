#pragma once

#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>

namespace thirdai::automl::deployment {

using DataType = bolt::sequential_classifier::DataType;
using TemporalConfig = bolt::sequential_classifier::TemporalConfig;

using AutotunableTemporalRelationships =
    std::map<std::string,
             std::vector<std::variant<std::string, TemporalConfig>>>;

class TemporalConfigAutotuner {
 public:
  static inline auto autotune(
      const std::map<std::string, DataType>& data_types, uint32_t lookahead,
      AutotunableTemporalRelationships&& temporal_relationships) {
    std::map<std::string, std::vector<TemporalConfig>> temporal_configs;

    for (const auto& [tracking_key, trackables] : temporal_relationships) {
      for (const auto& trackable : trackables) {
        if (auto config = manualConfiguration(trackable)) {
          temporal_configs[tracking_key].push_back(config.value());

        } else {
          auto trackable_col = std::get<std::string>(trackable);
          auto autotuned_configs = autotunedConfigs(
              trackable_col, data_types.at(trackable_col), lookahead);
          temporal_configs[tracking_key].insert(
              temporal_configs[tracking_key].end(), autotuned_configs.begin(),
              autotuned_configs.end());
        }
      }
    }
    return temporal_configs;
  }

 private:
  static inline std::optional<TemporalConfig> manualConfiguration(
      const std::variant<std::string, TemporalConfig>& trackable) {
    if (std::holds_alternative<TemporalConfig>(trackable)) {
      return std::get<TemporalConfig>(trackable);
    }
    return std::nullopt;
  }

  static inline std::vector<TemporalConfig> autotunedConfigs(
      const std::string& trackable_col, DataType trackable_col_type,
      uint32_t lookahead) {
    if (trackable_col_type.isNumerical()) {
      uint32_t history_length =
          std::max(lookahead, static_cast<uint32_t>(1)) * 4;
      return {TemporalConfig::numerical(trackable_col, history_length)};
    }

    if (trackable_col_type.isCategorical()) {
      std::vector<uint32_t> window_sizes = {1, 2, 5, 10, 25};
      std::vector<TemporalConfig> configs;
      configs.reserve(window_sizes.size());
      for (uint32_t track_last_n : window_sizes) {
        configs.push_back(
            TemporalConfig::categorical(trackable_col, track_last_n));
      }
      return configs;
    }

    throw std::invalid_argument(
        trackable_col +
        " is neither numerical nor categorical. Only numerical and "
        "categorical columns can be tracked temporally.");
  }
};

struct OracleConfig {
  OracleConfig(std::map<std::string, DataType> data_types,
               AutotunableTemporalRelationships temporal_tracking_relationships,
               std::string target, std::string time_granularity = "d",
               uint32_t lookahead = 0)
      : data_types(std::move(data_types)),
        target(std::move(target)),
        time_granularity(
            dataset::stringToGranularity(std::move(time_granularity))),
        lookahead(lookahead),
        temporal_tracking_relationships(TemporalConfigAutotuner::autotune(
            this->data_types, this->lookahead,
            std::move(temporal_tracking_relationships))) {}

  std::map<std::string, DataType> data_types;
  std::string target;
  dataset::QuantityTrackingGranularity time_granularity;
  uint32_t lookahead;
  std::map<std::string, std::vector<TemporalConfig>>
      temporal_tracking_relationships;

 private:
  // Private constructor for Cereal.
  OracleConfig() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(data_types, target, time_granularity, lookahead,
            temporal_tracking_relationships);
  }
};

using OracleConfigPtr = std::shared_ptr<OracleConfig>;

}  // namespace thirdai::automl::deployment