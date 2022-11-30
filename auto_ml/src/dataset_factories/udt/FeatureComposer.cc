#include "FeatureComposer.h"

namespace thirdai::automl::data {

void FeatureComposer::verifyConfigIsValid(
    const UDTConfig& config,
    const TemporalRelationships& temporal_relationships) {
  if (temporal_relationships.count(config.target)) {
    throw std::invalid_argument(
        "The target column cannot be a temporal tracking key.");
  }

  for (const auto& [tracking_key_col_name, temporal_configs] :
       temporal_relationships) {
    if (!config.data_types.count(tracking_key_col_name)) {
      throw std::invalid_argument("The tracking key '" + tracking_key_col_name +
                                  "' is not found in data_types.");
    }

    if (!asCategorical(config.data_types.at(tracking_key_col_name))) {
      throw std::invalid_argument("Tracking keys must be categorical.");
    }

    if (asCategorical(config.data_types.at(tracking_key_col_name))->delimiter) {
      throw std::invalid_argument(
          "Tracking keys cannot have a delimiter; columns containing "
          "tracking keys must only have one value per row.");
    }

    for (const auto& temporal_config : temporal_configs) {
      if (!config.data_types.count(temporal_config.columnName())) {
        throw std::invalid_argument("The tracked column '" +
                                    temporal_config.columnName() +
                                    "' is not found in data_types.");
      }
    }
  }
}

std::vector<dataset::BlockPtr> FeatureComposer::makeNonTemporalFeatureBlocks(
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

    if (auto categorical = asCategorical(data_type)) {
      // if part of metadata
      if (vectors_map.count(col_name) && categorical->metadata_config) {
        blocks.push_back(dataset::MetadataCategoricalBlock::make(
            col_num, vectors_map.at(col_name), categorical->delimiter));
      }
      if (categorical->delimiter) {
        // 1. we treat multicategorical as a text block since all we really
        // want is unigrams of the "words" separated by some delimiter
        // 2. text hash range of MAXINT is fine since features are later
        // hashed into a range. In fact it may reduce hash collisions.
        blocks.push_back(dataset::UniGramTextBlock::make(
            col_num, /* dim= */ std::numeric_limits<uint32_t>::max(),
            *categorical->delimiter));
      } else {
        tabular_datatypes[col_num] = dataset::TabularDataType::Categorical;
      }
    }

    if (auto numerical = asNumerical(data_type)) {
      tabular_col_ranges[col_num] = numerical->range;
      tabular_col_bins[col_num] = getNumberOfBins(numerical->granularity);
      tabular_datatypes[col_num] = dataset::TabularDataType::Numeric;
    }

    if (auto text_meta = asText(data_type)) {
      if (text_meta->force_pairgram ||
          (text_meta->average_n_words &&
           text_meta->average_n_words <= text_pairgrams_word_limit)) {
        // text hash range of MAXINT is fine since features are later
        // hashed into a range. In fact it may reduce hash collisions.
        blocks.push_back(dataset::PairGramTextBlock::make(
            col_num, /* dim= */ std::numeric_limits<uint32_t>::max()));
      } else {
        blocks.push_back(dataset::UniGramTextBlock::make(
            col_num, /* dim= */ std::numeric_limits<uint32_t>::max()));
      }
    }

    if (asDate(data_type)) {
      blocks.push_back(dataset::DateBlock::make(col_num));
    }
  }

  // Blocks still need a hash range even though we later hash it into
  // a range because we still want to support block feature
  // concatenations.
  // TODO(Geordie): This is redundant, remove this later.
  // we always use tabular unigrams but add pairgrams on top of it if the
  // contextual_columns flag is true
  blocks.push_back(
      makeTabularHashFeaturesBlock(tabular_datatypes, tabular_col_ranges,
                                   column_numbers.getColumnNumToColNameMap(),
                                   contextual_columns, tabular_col_bins));

  return blocks;
}

std::vector<dataset::BlockPtr> FeatureComposer::makeTemporalFeatureBlocks(
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

uint32_t FeatureComposer::getNumberOfBins(const std::string& granularity_size) {
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

std::unordered_set<std::string> FeatureComposer::getNonTemporalColumns(
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

std::string FeatureComposer::getTimestampColumnName(const UDTConfig& config) {
  std::optional<std::string> timestamp;
  for (const auto& [col_name, data_type] : config.data_types) {
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

dataset::BlockPtr FeatureComposer::makeTemporalCategoricalBlock(
    uint32_t temporal_relationship_id, const UDTConfig& config,
    TemporalContext& context, const ColumnNumberMap& column_numbers,
    const TemporalConfig& temporal_config, const std::string& key_column,
    const std::string& timestamp_column, bool should_update_history,
    dataset::PreprocessedVectorsPtr vectors) {
  const auto& tracked_column = temporal_config.columnName();

  if (!asCategorical(config.data_types.at(tracked_column))) {
    throw std::invalid_argument(
        "temporal.categorical can only be used with categorical "
        "columns.");
  }

  auto tracked_meta = asCategorical(config.data_types.at(tracked_column));
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
      // item hash range of MAXINT is fine since features are later
      // hashed into a range. In fact it may reduce hash collisions.
      /* item_hash_range= */ std::numeric_limits<uint32_t>::max(),
      /* should_update_history= */ should_update_history,
      /* include_current_row= */ temporal_meta.include_current_row,
      /* item_col_delimiter= */ tracked_meta->delimiter,
      /* time_lag= */ time_lag, /* item_vectors= */ std::move(vectors));
}

dataset::BlockPtr FeatureComposer::makeTemporalNumericalBlock(
    uint32_t temporal_relationship_id, const UDTConfig& config,
    TemporalContext& context, const ColumnNumberMap& column_numbers,
    const TemporalConfig& temporal_config, const std::string& key_column,
    const std::string& timestamp_column, bool should_update_history) {
  const auto& tracked_column = temporal_config.columnName();

  if (!asNumerical(config.data_types.at(tracked_column))) {
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

dataset::TabularHashFeaturesPtr FeatureComposer::makeTabularHashFeaturesBlock(
    const std::vector<dataset::TabularDataType>& tabular_datatypes,
    const std::unordered_map<uint32_t, std::pair<double, double>>& col_ranges,
    const std::vector<std::string>& num_to_name, bool contextual_columns,
    std::unordered_map<uint32_t, uint32_t> col_num_bins) {
  auto tabular_metadata = std::make_shared<dataset::TabularMetadata>(
      tabular_datatypes, col_ranges, /* class_name_to_id= */ nullptr,
      /* column_names= */ num_to_name, /* col_to_num_bins= */ col_num_bins);

  // output range of MAXINT is fine since features are later
  // hashed into a range. In fact it may reduce hash collisions.
  return std::make_shared<dataset::TabularHashFeatures>(
      tabular_metadata,
      /* output_range= */ std::numeric_limits<uint32_t>::max(),
      /* with_pairgrams= */ contextual_columns);
}

dataset::ThreadSafeVocabularyPtr& FeatureComposer::vocabForColumn(
    ColumnVocabularies& column_vocabs, const std::string& column_name,
    uint32_t vocab_size) {
  if (!column_vocabs.count(column_name)) {
    column_vocabs[column_name] =
        dataset::ThreadSafeVocabulary::make(vocab_size);
  }
  return column_vocabs.at(column_name);
}
}  // namespace thirdai::automl::data