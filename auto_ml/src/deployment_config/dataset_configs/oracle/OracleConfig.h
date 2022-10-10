#pragma once

#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <_types/_uint32_t.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>

namespace thirdai::automl::deployment {

using OracleDataType = bolt::sequential_classifier::DataType;
using OracleDataTypeMap = std::map<std::string, OracleDataType>;

using OracleTemporalConfig = bolt::sequential_classifier::TemporalConfig;
using OracleTemporalConfigMap =
    std::map<std::string, std::vector<OracleTemporalConfig>>;

using OracleAutotunableTemporalRelationships =
    std::map<std::string,
             std::vector<std::variant<std::string, OracleTemporalConfig>>>;

struct OracleConfig {
  OracleConfig(
      OracleDataTypeMap data_types,
      OracleAutotunableTemporalRelationships temporal_tracking_relationships,
      std::string target, std::string time_granularity = "d",
      uint32_t lookahead = 0)
      : data_types(std::move(data_types)),
        target(std::move(target)),
        time_granularity(
            dataset::stringToGranularity(std::move(time_granularity))),
        lookahead(lookahead),
        temporal_tracking_relationships(autotuneTemporalConfigs(
            this->data_types, this->lookahead,
            std::move(temporal_tracking_relationships))) {}

  OracleDataTypeMap data_types;
  std::string target;
  dataset::QuantityTrackingGranularity time_granularity;
  uint32_t lookahead;
  OracleTemporalConfigMap temporal_tracking_relationships;

 private:
  static OracleTemporalConfigMap autotuneTemporalConfigs(
      const OracleDataTypeMap& data_types, uint32_t lookahead,
      OracleAutotunableTemporalRelationships&& temporal_relationships) {
    OracleTemporalConfigMap temporal_configs;

    for (const auto& [tracking_key, trackables] : temporal_relationships) {
      for (const auto& trackable : trackables) {
        if (auto config = manualConfiguration(trackable)) {
          temporal_configs[tracking_key].push_back(config.value());

        } else {
          auto trackable_col = std::get<std::string>(trackable);
          if (data_types.at(trackable_col).isNumerical()) {
            addAutotunedNumericalConfig(temporal_configs[tracking_key],
                                        trackable_col, lookahead);
          } else if (data_types.at(trackable_col).isCategorical()) {
            addAutotunedCategoricalConfig(temporal_configs[tracking_key],
                                          trackable_col);
          } else {
            throw std::invalid_argument(
                trackable_col +
                " is neither numerical nor categorical. Only numerical and "
                "categorical columns can be tracked temporally.");
          }
        }
      }
    }
    return temporal_configs;
  }

  static std::optional<OracleTemporalConfig> manualConfiguration(
      const std::variant<std::string, OracleTemporalConfig>& trackable) {
    if (std::holds_alternative<OracleTemporalConfig>(trackable)) {
      return std::get<OracleTemporalConfig>(trackable);
    }
    return std::nullopt;
  }

  static void addAutotunedNumericalConfig(
      std::vector<OracleTemporalConfig>& configs,
      const std::string& trackable_col, uint32_t lookahead) {
    uint32_t history_length = std::max(lookahead, static_cast<uint32_t>(1)) * 4;
    configs.push_back(
        OracleTemporalConfig::numerical(trackable_col, history_length));
  }

  static void addAutotunedCategoricalConfig(
      std::vector<OracleTemporalConfig>& configs,
      const std::string& trackable_col) {
    std::vector<uint32_t> window_sizes = {1, 2, 5, 10, 25};
    configs.reserve(configs.size() + window_sizes.size());
    for (uint32_t track_last_n : window_sizes) {
      configs.push_back(
          OracleTemporalConfig::categorical(trackable_col, track_last_n));
    }
  }

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