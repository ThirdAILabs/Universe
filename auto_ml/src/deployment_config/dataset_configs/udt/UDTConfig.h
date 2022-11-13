#pragma once

#include <cereal/access.hpp>
#include <cereal/types/string.hpp>
#include "DataTypes.h"
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>

namespace thirdai::automl::deployment {

struct UDTConfig {
  static constexpr uint32_t DEFAULT_HASH_RANGE = 100000;

  /**
   * data_types: mapping from column names (strings) to DataType objects,
   *   e.g. {"user_id_column": types.categorical(n_unique_classes=5)}
   *
   * temporal_tracking_relationships: mapping from column names (strings) to
   *   a list of other columns can be tracked against it (either strings or
   *   TemporalConfig objects).
   *   e.g. {"user_id": [
   *            "movie_id",
   *            temporal.categorical("movie_id", track_last_n=20)
   *        ]}
   *   When a TemporalConfig object is provided, UDT will track the column
   *   as manually configured. When a string is provided, the tracking
   *   configuration will be autotuned.
   *
   * target: column name of target variable.
   *
   * time_granularity: Either "daily"/"d", "weekly"/"w", "biweekly"/"b",
   *   or `"monthly"`/`"m"`. Interval of time that we are interested in.
   *   Temporal numerical features are grouped according to this time
   *   granularity. E.g. if time_granularity="w" and the numerical values
   *   on days 1 and 2 are 345.25 and 201.1 respectively, then UDT
   *   captures a single numerical value of 546.26 for the week instead of
   *   individual values for the two days. Defaults to "d" (daily).
   *
   * lookahead: How far in the future UDT has to predict. The given number
   *   is relative to the provided time_granularity.
   *   e.g. if time_granularity = "w" and lookahead = 5, then UDT learns to
   *   predict 5 weeks into the future.
   */
  UDTConfig(ColumnDataTypes data_types,
            UserProvidedTemporalRelationships temporal_tracking_relationships,
            std::string target, std::string time_granularity = "d",
            uint32_t lookahead = 0, char delimiter = ',')
      : data_types(std::move(data_types)),
        provided_relationships(std::move(temporal_tracking_relationships)),
        target(std::move(target)),
        time_granularity(
            dataset::stringToGranularity(std::move(time_granularity))),
        lookahead(lookahead),
        delimiter(delimiter) {}

  ColumnDataTypes data_types;
  UserProvidedTemporalRelationships provided_relationships;
  std::string target;
  dataset::QuantityTrackingGranularity time_granularity;
  uint32_t lookahead;
  char delimiter;

  uint32_t hash_range = DEFAULT_HASH_RANGE;

 private:
  // Private constructor for Cereal.
  UDTConfig() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(data_types, provided_relationships, target, time_granularity,
            lookahead, delimiter, hash_range);
  }
};

using UDTConfigPtr = std::shared_ptr<UDTConfig>;

}  // namespace thirdai::automl::deployment