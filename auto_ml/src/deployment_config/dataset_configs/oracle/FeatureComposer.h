#pragma once

#include "Aliases.h"
#include "OracleConfig.h"
#include "TemporalContext.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl::deployment {

class FeatureComposer {
 public:
  static std::vector<dataset::BlockPtr> makeSingleRowFeatureBlocks(
      const OracleConfig& config,
      const TemporalRelationships& temporal_relationships,
      const ColumnNumberMap& column_numbers, ColumnVocabularies& vocabularies) {
    std::vector<dataset::BlockPtr> blocks;

    auto trackable_oolumns = trackableColumns(temporal_relationships);

    // Order of column names and data types is always consistent because
    // data_types is an ordered map.
    for (const auto& [col_name, data_type] : config.data_types) {
      if (data_type.isCategorical()) {
        if (col_name != config.target &&
            (isTrackingKey(col_name, temporal_relationships) ||
             !trackable_oolumns.count(col_name))) {
          auto vocab_size = data_type.asCategorical().n_unique_classes;
          uint32_t col_num = column_numbers.at(col_name);

          blocks.push_back(dataset::StringLookupCategoricalBlock::make(
              /* col= */ col_num,
              /* vocab= */ vocabForColumn(vocabularies, col_name, vocab_size)));
        }
      }

      if (data_type.isNumerical()) {
        if (!trackable_oolumns.count(col_name)) {
          blocks.push_back(dataset::DenseArrayBlock::make(
              /* start_col= */ column_numbers.at(col_name), /* dim= */ 1));
        }
      }

      if (data_type.isText()) {
        auto text_meta = config.data_types.at(col_name).asText();
        auto col_num = column_numbers.at(col_name);
        if (text_meta.average_n_words && text_meta.average_n_words <= 15) {
          blocks.push_back(dataset::PairGramTextBlock::make(col_num));
        } else {
          blocks.push_back(dataset::UniGramTextBlock::make(col_num));
        }
      }

      if (data_type.isDate()) {
        blocks.push_back(dataset::DateBlock::make(
            /* col= */ column_numbers.at(col_name)));
      }
    }
    return blocks;
  }

  static std::vector<dataset::BlockPtr> makeTemporalFeatureBlocks(
      const OracleConfig& config,
      const TemporalRelationships& temporal_relationships,
      const ColumnNumberMap& column_numbers, ColumnVocabularies& vocabularies,
      TemporalContext& context, bool should_update_history) {
    std::vector<dataset::BlockPtr> blocks;

    auto timestamp = getTimestamp(config, temporal_relationships);

    /*
      Order of tracking keys is always consistent because
      temporal_tracking_relationships is an ordered map.
      Therefore, the order of ids is also consistent.
    */
    uint32_t id = 0;
    for (const auto& [tracking_key, temporal_configs] :
         temporal_relationships) {
      auto tracking_key_type = config.data_types.at(tracking_key);
      if (!tracking_key_type.isCategorical()) {
        throw std::invalid_argument("Tracking keys must be categorical.");
      }

      for (const auto& temporal_config : temporal_configs) {
        if (temporal_config.isCategorical()) {
          blocks.push_back(makeTemporalCategoricalBlock(
              config, context, vocabularies, id, column_numbers, tracking_key,
              timestamp, temporal_config, should_update_history));
        }

        if (temporal_config.isNumerical()) {
          blocks.push_back(makeTemporalNumericalBlock(
              config, context, id, column_numbers, tracking_key, timestamp,
              temporal_config, should_update_history));
        }

        id++;
      }
    }
    return blocks;
  }

 private:
  static std::unordered_set<std::string> trackableColumns(
      const TemporalRelationships& temporal_relationships) {
    std::unordered_set<std::string> trackables_set;
    for (const auto& [_, trackables] : temporal_relationships) {
      for (const auto& trackable : trackables) {
        trackables_set.insert(trackable.columnName());
      }
    }
    return trackables_set;
  }

  static bool isTrackingKey(
      const std::string& column_name,
      const TemporalRelationships& temporal_relationships) {
    return temporal_relationships.count(column_name);
  }

  static std::string getTimestamp(
      const OracleConfig& config,
      const TemporalRelationships& temporal_relationships) {
    std::optional<std::string> timestamp;
    if (!temporal_relationships.empty()) {
      for (const auto& [col_name, data_type] : config.data_types) {
        if (data_type.isDate()) {
          if (timestamp) {
            throw std::invalid_argument(
                "There can only be one timestamp column.");
          }
          timestamp = col_name;
        }
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
      const OracleConfig& config, TemporalContext& context,
      ColumnVocabularies& vocabs, uint32_t id,
      const ColumnNumberMap& column_numbers, const std::string& key_column,
      const std::string& timestamp_column,
      const TemporalConfig& temporal_config, bool should_update_history) {
    const auto& tracked_column = temporal_config.columnName();

    if (!config.data_types.at(tracked_column).isCategorical()) {
      throw std::invalid_argument(
          "temporal.categorical can only be used with categorical "
          "columns.");
    }

    auto key_vocab_size =
        config.data_types.at(key_column).asCategorical().n_unique_classes;
    auto tracked_vocab_size =
        config.data_types.at(tracked_column).asCategorical().n_unique_classes;
    auto temporal_meta = temporal_config.asCategorical();

    int64_t time_lag = config.lookahead;
    time_lag *= dataset::QuantityHistoryTracker::granularityToSeconds(
        config.time_granularity);

    return dataset::UserItemHistoryBlock::make(
        /* user_col= */ column_numbers.at(key_column),
        /* item_col= */ column_numbers.at(tracked_column),
        /* timestamp_col= */ column_numbers.at(timestamp_column),
        /* user_id_map= */ vocabForColumn(vocabs, key_column, key_vocab_size),
        /* item_id_map= */
        vocabForColumn(vocabs, tracked_column, tracked_vocab_size),
        /* records= */
        context.categoricalHistoryForId(id, /* n_users= */ key_vocab_size),
        /* track_last_n= */ temporal_meta.track_last_n,
        /* should_update_history= */ should_update_history,
        /* inlcude_current_row= */ temporal_meta.include_current_row,
        /* item_col_delimiter= */ std::nullopt,
        /* time_lag= */ time_lag);
  }

  static dataset::BlockPtr makeTemporalNumericalBlock(
      const OracleConfig& config, TemporalContext& context, uint32_t id,
      const ColumnNumberMap& column_numbers, const std::string& key_column,
      const std::string& timestamp_column,
      const TemporalConfig& temporal_config, bool should_update_history) {
    const auto& tracked_column = temporal_config.columnName();

    if (!config.data_types.at(tracked_column).isNumerical()) {
      throw std::invalid_argument(
          "temporal.numerical can only be used with numerical columns.");
    }

    auto temporal_meta = temporal_config.asNumerical();

    auto numerical_history = context.numericalHistoryForId(
        /* id= */ id,
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
