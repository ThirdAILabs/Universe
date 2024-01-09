#include "TabularBlockComposer.h"
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Sequence.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/blocks/UserCountHistory.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>

namespace thirdai::automl {

namespace {

std::unordered_set<std::string> getNonTemporalColumnNames(
    const ColumnDataTypes& data_types,
    const TemporalRelationships& temporal_relationships);

std::string getTimestampColumnName(const ColumnDataTypes& input_data_types);

dataset::BlockPtr makeTemporalCategoricalBlock(
    uint32_t temporal_relationship_id, const ColumnDataTypes& input_data_types,
    TemporalContext& temporal_context, const TemporalConfig& temporal_config,
    const std::string& key_column, const std::string& timestamp_column,
    bool should_update_history, dataset::PreprocessedVectorsPtr vectors,
    dataset::QuantityTrackingGranularity time_granularity, uint32_t lookahead);

dataset::BlockPtr makeTemporalNumericalBlock(
    uint32_t temporal_relationship_id, const ColumnDataTypes& input_data_types,
    TemporalContext& temporal_context, const TemporalConfig& temporal_config,
    const std::string& key_column, const std::string& timestamp_column,
    bool should_update_history,
    dataset::QuantityTrackingGranularity time_granularity, uint32_t lookahead);

}  // namespace

std::vector<dataset::BlockPtr> makeTabularInputBlocks(
    const ColumnDataTypes& data_types,
    const std::set<std::string>& label_col_names,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    TemporalContext& temporal_context, bool should_update_history,
    const TabularOptions& options) {
  std::vector<dataset::BlockPtr> blocks =
      makeNonTemporalInputBlocks(data_types, label_col_names,
                                 temporal_relationships, vectors_map, options);

  if (temporal_relationships.empty()) {
    return blocks;
  }

  auto temporal_feature_blocks =
      makeTemporalInputBlocks(data_types, temporal_relationships, vectors_map,
                              temporal_context, should_update_history, options);

  blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                temporal_feature_blocks.end());
  return blocks;
}

std::vector<dataset::BlockPtr> makeNonTemporalInputBlocks(
    const ColumnDataTypes& data_types,
    const std::set<std::string>& label_col_names,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map, const TabularOptions& options) {
  std::vector<dataset::BlockPtr> blocks;

  auto non_temporal_columns =
      getNonTemporalColumnNames(data_types, temporal_relationships);

  std::vector<dataset::TabularColumn> tabular_columns;

  for (const auto& [col_name, data_type] : data_types) {
    if (!non_temporal_columns.count(col_name) ||
        label_col_names.count(col_name)) {
      continue;
    }

    if (auto categorical = asCategorical(data_type)) {
      // if part of metadata
      if (vectors_map.count(col_name) && categorical->metadata_config) {
        blocks.push_back(dataset::MetadataCategoricalBlock::make(
            col_name, vectors_map.at(col_name), categorical->delimiter));
      }
      if (categorical->delimiter) {
        // 1. we treat multicategorical as a text block since all we really
        // want is unigrams of the "words" separated by some delimiter
        // 2. text hash range of MAXINT is fine since features are later
        // hashed into a range. In fact it may reduce hash collisions.
        blocks.push_back(dataset::TextBlock::make(
            /* col = */ col_name,
            /* tokenizer = */
            dataset::NaiveSplitTokenizer::make(*categorical->delimiter),
            /* encoder = */ dataset::NGramEncoder::make(/* n = */ 1),
            /* lowercase = */ false,
            /* dim= */ std::numeric_limits<uint32_t>::max()));
      } else {
        tabular_columns.push_back(dataset::TabularColumn::Categorical(
            /* identifier= */ col_name));
      }
    }

    if (auto numerical = asNumerical(data_type)) {
      // tabular_datatypes.size() is the index of the next tabular data type.
      tabular_columns.push_back(dataset::TabularColumn::Numeric(
          /* identifier= */ col_name, /* range= */ numerical->range,
          /* num_bins= */ numerical->numBins()));
    }

    if (auto text_meta = asText(data_type)) {
      blocks.push_back(dataset::TextBlock::make(
          col_name, text_meta->tokenizer, text_meta->encoder,
          /* lowercase = */ text_meta->lowercase,
          /* dim = */ std::numeric_limits<uint32_t>::max(),
          /* cleaner = */ text_meta->cleaner));
    }

    if (asDate(data_type)) {
      blocks.push_back(dataset::DateBlock::make(col_name));
    }

    if (auto sequence = asSequence(data_type)) {
      blocks.push_back(dataset::SequenceBlock::make(
          col_name, /* delimiter= */ sequence->delimiter,
          /* dim= */ std::numeric_limits<uint32_t>::max()));
    }
  }

  // Blocks still need a hash range even though we later hash it into
  // a range because we still want to support block feature
  // concatenations.
  // TODO(Geordie): This is redundant, remove this later.
  // we always use tabular unigrams but add pairgrams on top of it if the
  // contextual_columns flag is true
  blocks.push_back(std::make_shared<dataset::TabularHashFeatures>(
      /* columns= */ tabular_columns,
      /* output_range= */ std::numeric_limits<uint32_t>::max(),
      /* with_pairgrams= */ options.contextual_columns));

  return blocks;
}

std::vector<dataset::BlockPtr> makeTemporalInputBlocks(
    const ColumnDataTypes& data_types,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    TemporalContext& temporal_context, bool should_update_history,
    const TabularOptions& options) {
  std::vector<dataset::BlockPtr> blocks;

  auto timestamp_col_name = getTimestampColumnName(data_types);

  dataset::QuantityTrackingGranularity granularity =
      dataset::stringToGranularity(options.time_granularity);

  /*
    Order of tracking keys is always consistent because
    temporal_tracking_relationships is an ordered map.
    Therefore, the order of ids is also consistent.
  */
  uint32_t temporal_relationship_id = 0;
  for (const auto& [tracking_key_col_name, temporal_configs] :
       temporal_relationships) {
    if (!data_types.count(tracking_key_col_name)) {
      throw std::invalid_argument("The tracking key '" + tracking_key_col_name +
                                  "' is not found in data_types.");
    }
    for (const auto& temporal_config : temporal_configs) {
      if (temporal_config.isCategorical()) {
        blocks.push_back(makeTemporalCategoricalBlock(
            temporal_relationship_id, data_types, temporal_context,
            temporal_config, tracking_key_col_name, timestamp_col_name,
            should_update_history,
            /* vectors= */ nullptr, granularity, options.lookahead));
        if (vectors_map.count(temporal_config.columnName()) &&
            temporal_config.asCategorical().use_metadata) {
          blocks.push_back(makeTemporalCategoricalBlock(
              temporal_relationship_id, data_types, temporal_context,
              temporal_config, tracking_key_col_name, timestamp_col_name,
              should_update_history,
              vectors_map.at(temporal_config.columnName()), granularity,
              options.lookahead));
        }
      }

      if (temporal_config.isNumerical()) {
        blocks.push_back(makeTemporalNumericalBlock(
            temporal_relationship_id, data_types, temporal_context,
            temporal_config, tracking_key_col_name, timestamp_col_name,
            should_update_history, granularity, options.lookahead));
      }

      temporal_relationship_id++;
    }
  }
  return blocks;
}

namespace {

std::unordered_set<std::string> getNonTemporalColumnNames(
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

std::string getTimestampColumnName(const ColumnDataTypes& input_data_types) {
  std::optional<std::string> timestamp;
  for (const auto& [col_name, data_type] : input_data_types) {
    if (asDate(data_type)) {
      if (timestamp) {
        throw std::invalid_argument("There can only be one timestamp column.");
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

dataset::BlockPtr makeTemporalCategoricalBlock(
    uint32_t temporal_relationship_id, const ColumnDataTypes& input_data_types,
    TemporalContext& temporal_context, const TemporalConfig& temporal_config,
    const std::string& key_column, const std::string& timestamp_column,
    bool should_update_history, dataset::PreprocessedVectorsPtr vectors,
    dataset::QuantityTrackingGranularity time_granularity, uint32_t lookahead) {
  const auto& tracked_column = temporal_config.columnName();

  if (!input_data_types.count(tracked_column)) {
    throw std::invalid_argument("The tracked column '" + tracked_column +
                                "' is not found in data_types.");
  }
  if (!asCategorical(input_data_types.at(tracked_column))) {
    throw std::invalid_argument(
        "temporal.categorical can only be used with categorical "
        "columns.");
  }

  auto tracked_meta = asCategorical(input_data_types.at(tracked_column));
  auto temporal_meta = temporal_config.asCategorical();

  int64_t time_lag = lookahead;
  time_lag *=
      dataset::QuantityHistoryTracker::granularityToSeconds(time_granularity);

  return dataset::UserItemHistoryBlock::make(
      /* user_col= */ key_column,
      /* item_col= */ tracked_column,
      /* timestamp_col= */ timestamp_column,
      /* records= */
      temporal_context.categoricalHistoryForId(temporal_relationship_id),
      /* track_last_n= */ temporal_meta.track_last_n,
      // item hash range of MAXINT is fine since features are later
      // hashed into a range. In fact it may reduce hash collisions.
      /* item_hash_range= */ std::numeric_limits<uint32_t>::max(),
      /* should_update_history= */ should_update_history,
      /* include_current_row= */ temporal_meta.include_current_row,
      /* item_col_delimiter= */ tracked_meta->delimiter,
      /* time_lag= */ time_lag, /* item_vectors= */ std::move(vectors));
}

dataset::BlockPtr makeTemporalNumericalBlock(
    uint32_t temporal_relationship_id, const ColumnDataTypes& input_data_types,
    TemporalContext& temporal_context, const TemporalConfig& temporal_config,
    const std::string& key_column, const std::string& timestamp_column,
    bool should_update_history,
    dataset::QuantityTrackingGranularity time_granularity, uint32_t lookahead) {
  const auto& tracked_column = temporal_config.columnName();

  if (!asNumerical(input_data_types.at(tracked_column))) {
    throw std::invalid_argument(
        "temporal.numerical can only be used with numerical columns.");
  }

  auto temporal_meta = temporal_config.asNumerical();

  auto numerical_history = temporal_context.numericalHistoryForId(
      /* id= */ temporal_relationship_id,
      /* lookahead= */ lookahead,
      /* history_length= */ temporal_meta.history_length,
      /* time_granularity= */ time_granularity);

  return dataset::UserCountHistoryBlock::make(
      /* user_col= */ key_column,
      /* count_col= */ tracked_column,
      /* timestamp_col= */ timestamp_column,
      /* history= */ numerical_history,
      /* should_update_history= */ should_update_history,
      /* include_current_row= */ temporal_meta.include_current_row);
}

}  // namespace

}  // namespace thirdai::automl