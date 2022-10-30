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
#include <string>

namespace thirdai::automl::deployment {

class TemporalRelationshipsAutotuner {
 public:
  /**
   * provided_relationships is a mapping from column names (strings) to
   *   a list of other columns can be tracked against it (either strings or
   *   TemporalConfig objects).
   *   e.g. {"user_id": [
   *            "movie_id",
   *            temporal.categorical("movie_id", track_last_n=20)
   *        ]}
   *
   *   The purpose of the autotune() method is to convert this into a mapping
   *   from column names to a list of TemporalConfig objects by replacing
   *   the strings with suitable TemporalConfig object(s) based on heuristics
   *   that gave good empirical results.
   */
  static inline TemporalRelationships autotune(
      const ColumnDataTypes& data_types,
      const UserProvidedTemporalRelationships& provided_relationships,
      uint32_t lookahead) {
    TemporalRelationships configs;

    for (const auto& [key, tracked_items] : provided_relationships) {
      for (const auto& tracked_item : tracked_items) {
        if (std::holds_alternative<std::string>(tracked_item)) {
          auto tracked_col_name = std::get<std::string>(tracked_item);

          if (data_types.at(tracked_col_name).isNumerical()) {
            makeNumericalConfigs(configs[key], tracked_col_name, lookahead);

          } else if (data_types.at(tracked_col_name).isCategorical()) {
            makeCategoricalConfigs(configs[key], tracked_col_name);

          } else {
            throw std::invalid_argument(
                tracked_col_name +
                " is neither numerical nor categorical. Only numerical and "
                "categorical columns can be tracked temporally.");
          }
        } else {
          configs[key].push_back(std::get<TemporalConfig>(tracked_item));
        }
      }
    }
    return configs;
  }

 private:
  static void makeNumericalConfigs(std::vector<TemporalConfig>& configs,
                                   const std::string& tracked_col,
                                   uint32_t lookahead) {
    uint32_t history_length = std::max<uint32_t>(lookahead, 1) * 4;
    configs.push_back(TemporalConfig::numerical(tracked_col, history_length));
  }

  static void makeCategoricalConfigs(std::vector<TemporalConfig>& configs,
                                     const std::string& trackable_col) {
    /*
      For each window size W, there is a segment in the feature vector that
      represents a set of up to W items.
      We have multiple tracking window sizes so the model is aware of short-,
      medium-, and long-term triends.
    */
    std::vector<uint32_t> window_sizes = {1, 2, 5, 10, 25};
    configs.reserve(configs.size() + window_sizes.size());
    for (uint32_t track_last_n : window_sizes) {
      configs.push_back(TemporalConfig::categorical(
          trackable_col, track_last_n, /* include_current_row= */ false,
          // Use metadata in medium-sized window.
          /* use_metadata= */ track_last_n == window_sizes[2]));
    }
  }
};

}  // namespace thirdai::automl::deployment