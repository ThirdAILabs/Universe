#pragma once

#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/Aliases.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>

namespace thirdai::automl::deployment {

class TemporalRelationshipsAutotuner {
 public:
  static inline TemporalRelationships autotune(
      const ColumnDataTypes& data_types,
      const UserProvidedTemporalRelationships& provided_relationships,
      uint32_t lookahead) {
    TemporalRelationships temporal_relationships;

    for (const auto& [tracking_key, trackables] : provided_relationships) {
      for (const auto& trackable : trackables) {
        if (auto config = manualConfiguration(trackable)) {
          temporal_relationships[tracking_key].push_back(config.value());

        } else {
          auto trackable_col = std::get<std::string>(trackable);
          if (data_types.at(trackable_col).isNumerical()) {
            addAutotunedNumericalConfig(temporal_relationships[tracking_key],
                                        trackable_col, lookahead);
          } else if (data_types.at(trackable_col).isCategorical()) {
            addAutotunedCategoricalConfig(temporal_relationships[tracking_key],
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
    return temporal_relationships;
  }

 private:
  static std::optional<TemporalConfig> manualConfiguration(
      const std::variant<std::string, TemporalConfig>& trackable) {
    if (std::holds_alternative<TemporalConfig>(trackable)) {
      return std::get<TemporalConfig>(trackable);
    }
    return std::nullopt;
  }

  static void addAutotunedNumericalConfig(std::vector<TemporalConfig>& configs,
                                          const std::string& trackable_col,
                                          uint32_t lookahead) {
    uint32_t history_length = std::max(lookahead, static_cast<uint32_t>(1)) * 4;
    configs.push_back(TemporalConfig::numerical(trackable_col, history_length));
  }

  static void addAutotunedCategoricalConfig(
      std::vector<TemporalConfig>& configs, const std::string& trackable_col) {
    std::vector<uint32_t> window_sizes = {1, 2, 5, 10, 25};
    configs.reserve(configs.size() + window_sizes.size());
    for (uint32_t track_last_n : window_sizes) {
      configs.push_back(
          TemporalConfig::categorical(trackable_col, track_last_n));
    }
  }
};

}  // namespace thirdai::automl::deployment