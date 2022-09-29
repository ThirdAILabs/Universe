#pragma once

#include <cereal/access.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include "ConstructorUtilityTypes.h"
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <map>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::bolt::sequential_classifier {

struct SequentialClassifierConfig {
  std::map<std::string, DataType> data_types;
  std::map<std::string, std::vector<TemporalConfig>>
      temporal_tracking_relationships;
  std::string target;
  dataset::QuantityTrackingGranularity time_granularity;
  uint32_t lookahead;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(data_types, temporal_tracking_relationships, target,
            time_granularity, lookahead);
  }
};

}  // namespace thirdai::bolt::sequential_classifier