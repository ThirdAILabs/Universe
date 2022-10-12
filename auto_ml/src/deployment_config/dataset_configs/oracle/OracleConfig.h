#pragma once

#include <cereal/access.hpp>
#include <cereal/types/string.hpp>
#include "Aliases.h"
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

struct OracleConfig {
  OracleConfig(
      ColumnDataTypes data_types,
      UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target, std::string time_granularity = "d",
      uint32_t lookahead = 0)
      : data_types(std::move(data_types)),
        target(std::move(target)),
        time_granularity(
            dataset::stringToGranularity(std::move(time_granularity))),
        lookahead(lookahead),
        provided_relationships(std::move(temporal_tracking_relationships)) {}

  ColumnDataTypes data_types;
  std::string target;
  dataset::QuantityTrackingGranularity time_granularity;
  uint32_t lookahead;
  UserProvidedTemporalRelationships provided_relationships;

 private:
  // Private constructor for Cereal.
  OracleConfig() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(data_types, target, time_granularity, lookahead,
            provided_relationships);
  }
};

using OracleConfigPtr = std::shared_ptr<OracleConfig>;

}  // namespace thirdai::automl::deployment