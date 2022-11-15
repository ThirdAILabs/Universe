#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include "ColumnNumberMap.h"
#include "DataTypes.h"
#include "TemporalContext.h"
#include "UDTConfig.h"
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
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
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::deployment {

using PreprocessedVectorsMap =
    std::unordered_map<std::string, dataset::PreprocessedVectorsPtr>;
using ColumnVocabularies =
    std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr>;

class FeatureComposer {
 public:
  static void verifyConfigIsValid(
      const UDTConfig& config,
      const TemporalRelationships& temporal_relationships) {
    if (temporal_relationships.count(config.target)) {
      throw std::invalid_argument(
          "The target column cannot be a temporal tracking key.");
    }

    for (const auto& [tracking_key_col_name, temporal_configs] :
         temporal_relationships) {
      if (!config.data_types.at(tracking_key_col_name).isCategorical()) {
        throw std::invalid_argument("Tracking keys must be categorical.");
      }

      if (config.data_types.at(tracking_key_col_name)
              .asCategorical()
              .delimiter) {
        throw std::invalid_argument(
            "Tracking keys cannot have a delimiter; columns containing "
            "tracking keys must only have one value per row.");
      }
    }
  }

  static std::vector<dataset::BlockPtr> makeNonTemporalFeatureBlocks(
      const UDTConfig& config,
      const TemporalRelationships& temporal_relationships,
      const ColumnNumberMap& column_numbers,
      const PreprocessedVectorsMap& vectors_map,
      uint32_t text_pairgrams_word_limit, bool contextual_columns) {
    std::vector<dataset::BlockPtr> blocks;

    auto non_temporal_columns =
        getNonTemporalColumns(config.data_types, temporal_relationships);

    std::vector<dataset::TabularDataType> tabular_datatypes(
        column_numbers.numCols(), dataset::TabularDataType::Ignore);

    std::unordered_map<uint32_t, std::pair<double, double>> tabular_col_ranges;
    std::unordered_map<uint32_t, uint32_t> tabular_col_bins;

    /*
      Order of column names and data types is always consistent because
      data_types is an ordered map. Thus, the order of the input blocks
      remains consistent and so does the order of the vector segments.
    */
    for (const auto& [col_name, data_type] : config.data_types) {
      if (!non_temporal_columns.count(col_name) || col_name == config.target) {
        continue;
      }

      uint32_t col_num = column_numbers.at(col_name);

      if (data_type.isCategorical()) {
        auto categorical = data_type.asCategorical();
        // if has metadata
        if (vectors_map.count(col_name) && categorical.metadata_config) {
          blocks.push_back(dataset::MetadataCategoricalBlock::make(
              col_num, vectors_map.at(col_name), categorical.delimiter));
        }
        if (categorical.delimiter) {
          blocks.push_back(dataset::UniGramTextBlock::make(
              col_num, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
              *categorical.delimiter));
        } else {
          tabular_datatypes[col_num] = dataset::TabularDataType::Categorical;
        }
      }

      if (data_type.isNumerical()) {
        tabular_col_ranges[col_num] = data_type.asNumerical().range;
        tabular_col_bins[col_num] =
            getNumberOfBins(data_type.asNumerical().granularity);
        tabular_datatypes[col_num] = dataset::TabularDataType::Numeric;
      }

      if (data_type.isText()) {
        auto text_meta = data_type.asText();
        if (text_meta.force_pairgram ||
            (text_meta.average_n_words &&
             text_meta.average_n_words <= text_pairgrams_word_limit)) {
          blocks.push_back(dataset::PairGramTextBlock::make(col_num));
        } else {
          blocks.push_back(
              dataset::UniGramTextBlock::make(col_num, text_meta.dim));
        }
      }

      if (data_type.isDate()) {
        blocks.push_back(dataset::DateBlock::make(col_num));
      }
    }

    // we always use tabular unigrams but add pairgrams on top of it if the
    // contextual_columns flag is true
    blocks.push_back(makeTabularHashFeaturesBlock(
        tabular_datatypes, tabular_col_ranges,
        column_numbers.getColumnNumToColNameMap(), contextual_columns,
        tabular_col_bins, config.hash_range));

    return blocks;
  }

  static std::vector<dataset::BlockPtr> makeTemporalFeatureBlocks(
      const UDTConfig& config,
      const TemporalRelationships& temporal_relationships,
      const ColumnNumberMap& column_numbers,
      const PreprocessedVectorsMap& vectors_map, TemporalContext& context,
      bool should_update_history) {
    std::vector<dataset::BlockPtr> blocks;

    auto timestamp_col_name = getTimestampColumnName(config);

    /*
      Order of tracking keys is always consistent because
      temporal_tracking_relationships is an ordered map.
      Therefore, the order of ids is also consistent.
    */
    uint32_t temporal_relationship_id = 0;
    for (const auto& [tracking_key_col_name, temporal_configs] :
         temporal_relationships) {
      for (const auto& temporal_config : temporal_configs) {
        if (temporal_config.isCategorical()) {
          blocks.push_back(makeTemporalCategoricalBlock(
              temporal_relationship_id, config, context, column_numbers,
              temporal_config, tracking_key_col_name, timestamp_col_name,
              should_update_history,
              /* vectors= */ nullptr));
          if (vectors_map.count(temporal_config.columnName()) &&
              temporal_config.asCategorical().use_metadata) {
            blocks.push_back(makeTemporalCategoricalBlock(
                temporal_relationship_id, config, context, column_numbers,
                temporal_config, tracking_key_col_name, timestamp_col_name,
                should_update_history,
                vectors_map.at(temporal_config.columnName())));
          }
        }

        if (temporal_config.isNumerical()) {
          blocks.push_back(makeTemporalNumericalBlock(
              temporal_relationship_id, config, context, column_numbers,
              temporal_config, tracking_key_col_name, timestamp_col_name,
              should_update_history));
        }

        temporal_relationship_id++;
      }
    }
    return blocks;
  }

 private:
  static uint32_t getNumberOfBins(const std::string& granularity_size) {
    auto lower_size = utils::lower(granularity_size);
    if (lower_size == "xs" || lower_size == "extrasmall") {
      return 10;
    }
    if (lower_size == "s" || lower_size == "small") {
      return 75;
    }
    if (lower_size == "m" || lower_size == "medium") {
      return 300;
    }
    if (lower_size == "l" || lower_size == "large") {
      return 1000;
    }
    if (lower_size == "xl" || lower_size == "extralarge") {
      return 3000;
    }
    throw std::invalid_argument("Invalid numerical granularity \"" +
                                granularity_size +
                                "\". Choose one of \"extrasmall\"/\"xs\", "
                                "\"small\"/\"s\", \"medium\"/\"m\", "
                                "\"large\"/\"l\", or \"extralarge\"/\"xl\".");
  }
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
      const TemporalRelationships& temporal_relationships) {
    std::unordered_set<std::string> non_temporal_columns;
    for (const auto& [col_name, _] : data_types) {
      non_temporal_columns.insert(col_name);
    }
    for (const auto& [_, temporal_configs] : temporal_relationships) {
      for (const auto& temporal_config : temporal_configs) {
        if (non_temporal_columns.count(temporal_config.columnName())) {
          non_temporal_columns.erase(temporal_config.columnName());
        }
      }
    }
    // So far, non_temporal_columns contains column that are not tracked.
    // We now take the union with the set of tracking key columns
    for (const auto& [tracking_key_col_name, _] : temporal_relationships) {
      non_temporal_columns.insert(tracking_key_col_name);
    }
    return non_temporal_columns;
  }

  static bool isTrackingKey(
      const std::string& column_name,
      const TemporalRelationships& temporal_relationships) {
    return temporal_relationships.count(column_name);
  }

  static std::string getTimestampColumnName(const UDTConfig& config) {
    std::optional<std::string> timestamp;
    for (const auto& [col_name, data_type] : config.data_types) {
      if (data_type.isDate()) {
        if (timestamp) {
          throw std::invalid_argument(
              "There can only be one timestamp column.");
        }
        timestamp = col_name;
      }
    }
    if (!timestamp) {
      throw std::invalid_argument(
          "There has to be a timestamp column in order to use temporal "
          "tracking relationships.");
    }
    return *timestamp;
  }

  static dataset::BlockPtr makeTemporalCategoricalBlock(
      uint32_t temporal_relationship_id, const UDTConfig& config,
      TemporalContext& context, const ColumnNumberMap& column_numbers,
      const TemporalConfig& temporal_config, const std::string& key_column,
      const std::string& timestamp_column, bool should_update_history,
      dataset::PreprocessedVectorsPtr vectors) {
    const auto& tracked_column = temporal_config.columnName();

    if (!config.data_types.at(tracked_column).isCategorical()) {
      throw std::invalid_argument(
          "temporal.categorical can only be used with categorical "
          "columns.");
    }

    auto tracked_meta = config.data_types.at(tracked_column).asCategorical();
    auto temporal_meta = temporal_config.asCategorical();

    int64_t time_lag = config.lookahead;
    time_lag *= dataset::QuantityHistoryTracker::granularityToSeconds(
        config.time_granularity);

    return dataset::UserItemHistoryBlock::make(
        /* user_col= */ column_numbers.at(key_column),
        /* item_col= */ column_numbers.at(tracked_column),
        /* timestamp_col= */ column_numbers.at(timestamp_column),
        /* records= */
        context.categoricalHistoryForId(temporal_relationship_id),
        /* track_last_n= */ temporal_meta.track_last_n,
        /* item_hash_range= */ config.hash_range,
        /* should_update_history= */ should_update_history,
        /* include_current_row= */ temporal_meta.include_current_row,
        /* item_col_delimiter= */ tracked_meta.delimiter,
        /* time_lag= */ time_lag, /* item_vectors= */ std::move(vectors));
  }

  static dataset::BlockPtr makeTemporalNumericalBlock(
      uint32_t temporal_relationship_id, const UDTConfig& config,
      TemporalContext& context, const ColumnNumberMap& column_numbers,
      const TemporalConfig& temporal_config, const std::string& key_column,
      const std::string& timestamp_column, bool should_update_history) {
    const auto& tracked_column = temporal_config.columnName();

    if (!config.data_types.at(tracked_column).isNumerical()) {
      throw std::invalid_argument(
          "temporal.numerical can only be used with numerical columns.");
    }

    auto temporal_meta = temporal_config.asNumerical();

    auto numerical_history = context.numericalHistoryForId(
        /* id= */ temporal_relationship_id,
        /* lookahead= */ config.lookahead,
        /* history_length= */ temporal_meta.history_length,
        /* time_granularity= */ config.time_granularity);

    return dataset::UserCountHistoryBlock::make(
        /* user_col= */ column_numbers.at(key_column),
        /* count_col= */ column_numbers.at(tracked_column),
        /* timestamp_col= */ column_numbers.at(timestamp_column),
        /* history= */ numerical_history,
        /* should_update_history= */ should_update_history,
        /* include_current_row= */ temporal_meta.include_current_row);
  }

  static dataset::TabularHashFeaturesPtr makeTabularHashFeaturesBlock(
      const std::vector<dataset::TabularDataType>& tabular_datatypes,
      const std::unordered_map<uint32_t, std::pair<double, double>>& col_ranges,
      const std::vector<std::string>& num_to_name, bool contextual_columns,
      std::unordered_map<uint32_t, uint32_t> col_num_bins,
      uint32_t output_range) {
    auto tabular_metadata = std::make_shared<dataset::TabularMetadata>(
        tabular_datatypes, col_ranges, /* class_name_to_id= */ nullptr,
        /* column_names= */ num_to_name, /* col_to_num_bins= */ col_num_bins);

    return std::make_shared<dataset::TabularHashFeatures>(
        tabular_metadata, output_range,
        /* with_pairgrams= */ contextual_columns);
  }

  static dataset::ThreadSafeVocabularyPtr& vocabForColumn(
      ColumnVocabularies& column_vocabs, const std::string& column_name,
      uint32_t vocab_size) {
    if (!column_vocabs.count(column_name)) {
      column_vocabs[column_name] =
          dataset::ThreadSafeVocabulary::make(vocab_size);
    }
    return column_vocabs.at(column_name);
  }
};

}  // namespace thirdai::automl::deployment
