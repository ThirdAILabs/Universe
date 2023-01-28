#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include "DataTypes.h"
#include "TemporalContext.h"
#include "UDTConfig.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::data {

using PreprocessedVectorsMap =
    std::unordered_map<std::string, dataset::PreprocessedVectorsPtr>;
using ColumnVocabularies =
    std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr>;

class FeatureComposer {
 public:
  static void verifyConfigIsValid(
      const ColumnDataTypes& data_types, const std::string& target,
      const TemporalRelationships& temporal_relationships);

  static std::vector<dataset::BlockPtr> makeNonTemporalFeatureBlocks(
      const ColumnDataTypes& data_types, const std::string& target,
      const TemporalRelationships& temporal_relationships,
      const PreprocessedVectorsMap& vectors_map,
      uint32_t text_pairgrams_word_limit, bool contextual_columns);

  static std::vector<dataset::BlockPtr> makeTemporalFeatureBlocks(
      const UDTConfig& config,
      const TemporalRelationships& temporal_relationships,
      const PreprocessedVectorsMap& vectors_map, TemporalContext& context,
      bool should_update_history);

 private:
  static uint32_t getNumberOfBins(const std::string& granularity_size);
  /**
   * A column is encoded in a non-temporal way when it fulfils any
   * of the following:
   *  1. It is a temporal tracking key; columns are tracked against
   *     this column. E.g. if we track the movies that a user has
   *     watched, then the user column must be encoded in a non-
   *     temporal way.
   *  2. It is not a tracked column. E.g. if we track the movies
   *     that a user has watched, then the movie column is a
   *     tracked column.
   */
  static std::unordered_set<std::string> getNonTemporalColumns(
      const ColumnDataTypes& data_types,
      const TemporalRelationships& temporal_relationships);

  static bool isTrackingKey(
      const std::string& column_name,
      const TemporalRelationships& temporal_relationships) {
    return temporal_relationships.count(column_name);
  }

  static std::string getTimestampColumnName(const UDTConfig& config);

  static dataset::BlockPtr makeTemporalCategoricalBlock(
      uint32_t temporal_relationship_id, const UDTConfig& config,
      TemporalContext& context, const TemporalConfig& temporal_config,
      const std::string& key_column, const std::string& timestamp_column,
      bool should_update_history, dataset::PreprocessedVectorsPtr vectors);

  static dataset::BlockPtr makeTemporalNumericalBlock(
      uint32_t temporal_relationship_id, const UDTConfig& config,
      TemporalContext& context, const TemporalConfig& temporal_config,
      const std::string& key_column, const std::string& timestamp_column,
      bool should_update_history);

  static dataset::TabularHashFeaturesPtr makeTabularHashFeaturesBlock(
      const std::vector<dataset::TabularDataType>& tabular_datatypes,
      const std::unordered_map<uint32_t, std::pair<double, double>>& col_ranges,
      const std::vector<std::string>& column_names, bool contextual_columns,
      std::unordered_map<uint32_t, uint32_t> col_num_bins);
};

}  // namespace thirdai::automl::data
