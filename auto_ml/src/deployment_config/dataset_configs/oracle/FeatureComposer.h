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
#include <unordered_map>
#include <vector>

namespace thirdai::automl::deployment {

class FeatureComposer {
 public:
  static std::vector<dataset::BlockPtr> makeNonTemporalFeatureBlocks(
      const OracleConfig& config,
      const TemporalRelationships& temporal_relationships,
      const ColumnNumberMap& column_numbers, ColumnVocabularies& vocabularies,
      uint32_t use_text_pairgrams_if_under_n_words) {
    std::vector<dataset::BlockPtr> blocks;

    auto unknown_during_inference =
        getUnknownDuringInferenceColumns(temporal_relationships);

    // Order of column names and data types is always consistent because
    // data_types is an ordered map.
    for (const auto& [col_name, data_type] : config.data_types) {
      if (unknown_during_inference[col_name] || col_name == config.target) {
        continue;
      }

      uint32_t col_num = column_numbers.at(col_name);

      if (data_type.isCategorical()) {
        auto vocab_size = data_type.asCategorical().n_unique_classes;
        blocks.push_back(dataset::StringLookupCategoricalBlock::make(
            col_num, vocabForColumn(vocabularies, col_name, vocab_size)));
      }

      if (data_type.isNumerical()) {
        blocks.push_back(dataset::DenseArrayBlock::makeSingle(col_num));
      }

      if (data_type.isText()) {
        auto text_meta = data_type.asText();
        if (text_meta.average_n_words &&
            text_meta.average_n_words < use_text_pairgrams_if_under_n_words) {
          blocks.push_back(dataset::PairGramTextBlock::make(col_num));
        } else {
          blocks.push_back(dataset::UniGramTextBlock::make(col_num));
        }
      }

      if (data_type.isDate()) {
        blocks.push_back(dataset::DateBlock::make(col_num));
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

    auto timestamp_col_name = getTimestampColumnName(config);

    /*
      Order of tracking keys is always consistent because
      temporal_tracking_relationships is an ordered map.
      Therefore, the order of ids is also consistent.
    */
    uint32_t temporal_relationship_id = 0;
    for (const auto& [tracking_key_col_name, temporal_configs] :
         temporal_relationships) {
      if (!config.data_types.at(tracking_key_col_name).isCategorical()) {
        throw std::invalid_argument("Tracking keys must be categorical.");
      }

      for (const auto& temporal_config : temporal_configs) {
        if (temporal_config.isCategorical()) {
          blocks.push_back(makeTemporalCategoricalBlock(
              temporal_relationship_id, config, context, column_numbers,
              vocabularies, temporal_config, tracking_key_col_name,
              timestamp_col_name, should_update_history));
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
  static std::unordered_map<std::string, bool> getUnknownDuringInferenceColumns(
      const TemporalRelationships& temporal_relationships) {
    std::unordered_map<std::string, bool> is_unknown_during_inference;
    for (const auto& [_, temporal_configs] : temporal_relationships) {
      for (const auto& temporal_config : temporal_configs) {
        auto col_name = temporal_config.columnName();
        if (!is_unknown_during_inference.count(col_name)) {
          is_unknown_during_inference[col_name] = true;
        }
        is_unknown_during_inference[col_name] &=
            !temporal_config.includesCurrentRow();
      }
    }
    return is_unknown_during_inference;
  }

  static bool isTrackingKey(
      const std::string& column_name,
      const TemporalRelationships& temporal_relationships) {
    return temporal_relationships.count(column_name);
  }

  static std::string getTimestampColumnName(const OracleConfig& config) {
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
      uint32_t temporal_relationship_id, const OracleConfig& config,
      TemporalContext& context, const ColumnNumberMap& column_numbers,
      ColumnVocabularies& vocabs, const TemporalConfig& temporal_config,
      const std::string& key_column, const std::string& timestamp_column,
      bool should_update_history) {
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
        context.categoricalHistoryForId(temporal_relationship_id,
                                        /* n_users= */ key_vocab_size),
        /* track_last_n= */ temporal_meta.track_last_n,
        /* should_update_history= */ should_update_history,
        /* include_current_row= */ temporal_meta.include_current_row,
        /* item_col_delimiter= */ std::nullopt,
        /* time_lag= */ time_lag);
  }

  static dataset::BlockPtr makeTemporalNumericalBlock(
      uint32_t temporal_relationship_id, const OracleConfig& config,
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
