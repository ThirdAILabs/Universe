#include "TabularTransformations.h"
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/NumericalTemporal.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl {

// This represents the transformations and the output for a column in the input.
using TransformSeries =
    std::pair<std::vector<data::TransformationPtr>, std::string>;

TransformSeries text(const std::string& column_name,
                     const TextDataTypePtr& text,
                     size_t dim = std::numeric_limits<uint32_t>::max()) {
  std::string output = textOutputColumn(column_name);

  auto transformation = std::make_shared<data::TextTokenizer>(
      /* input_column= */ column_name, /* output_indices= */ output,
      /* output_values= */ std::nullopt, /* tokenizer= */ text->tokenizer,
      /* encoder= */ text->encoder, /* lowercase= */ text->lowercase,
      /* dim= */ dim);

  return {{transformation}, output};
}

TransformSeries categorical(const std::string& column_name,
                            const CategoricalDataTypePtr& categorical) {
  std::string output = categoricalOutputColumn(column_name);

  if (categorical->delimiter) {
    auto tok = dataset::NaiveSplitTokenizer::make(*categorical->delimiter);
    auto enc = dataset::NGramEncoder::make(/* n = */ 1);

    auto transformation = std::make_shared<data::TextTokenizer>(
        /* input_column= */ column_name, /* output_indices= */ output,
        /* output_values= */ std::nullopt, /* tokenizer= */ tok,
        /* encoder= */ enc, /* lowercase= */ false,
        /* dim= */ std::numeric_limits<uint32_t>::max());

    return {{transformation}, output};
  }

  auto transformation = std::make_shared<data::StringHash>(
      /* input_column_name= */ column_name, /* output_column_name= */ output);

  return {{transformation}, output};
}

TransformSeries sequence(const std::string& column_name,
                         const SequenceDataTypePtr& sequence) {
  std::string output = sequenceOutputColumn(column_name);

  auto hash = std::make_shared<data::StringHash>(
      column_name, column_name,
      /* hash_range= */ std::numeric_limits<uint32_t>::max(),
      /* delimiter= */ sequence->delimiter);

  auto transformation = std::make_shared<data::HashPositionTransform>(
      column_name, output,
      /* hash_range= */ std::numeric_limits<uint32_t>::max());

  return {{hash, transformation}, output};
}

TransformSeries date(const std::string& column_name,
                     const DateDataTypePtr& date) {
  (void)date;

  std::string output = dateOutputColumn(column_name);

  auto transformation = std::make_shared<data::Date>(
      /* input_column_name= */ column_name, /* output_column_name= */ output);

  return {{transformation}, output};
}

TransformSeries timestamp(const ColumnDataTypes& data_types) {
  std::optional<std::string> timestamp_column;
  for (const auto& [col_name, data_type] : data_types) {
    if (asDate(data_type)) {
      if (timestamp_column) {
        throw std::invalid_argument("There can only be one timestamp column.");
      }
      timestamp_column = col_name;
    }
  }
  if (!timestamp_column) {
    throw std::invalid_argument(
        "There has to be a timestamp column in order to use temporal "
        "tracking relationships.");
  }

  auto transformation = std::make_shared<data::StringToTimestamp>(
      /* input_column_name= */ *timestamp_column,
      /* output_column_name= */ TIMESTAMP_OUTPUT, /* format= */ "%Y-%m-%d");

  return {{transformation}, TIMESTAMP_OUTPUT};
}

MergedTransformSeries nonTemporalTransformations(
    const ColumnDataTypes& data_types, const TabularOptions& options) {
  std::vector<data::TransformationPtr> pipeline;
  std::vector<std::string> output_columns;

  std::vector<data::NumericalColumn> numerical_cols;
  std::vector<data::CategoricalColumn> categorical_cols;

  for (const auto& [name, data_type] : data_types) {
    if (auto text_type = asText(data_type)) {
      auto [transforms, output] = text(name, text_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    }

    if (auto cat_type = asCategorical(data_type)) {
      if (!cat_type->delimiter) {
        categorical_cols.push_back(data::CategoricalColumn(name));
      } else {
        auto [transforms, output] = categorical(name, cat_type);
        pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
        output_columns.push_back(output);
      }
    }

    if (auto numerical = asNumerical(data_type)) {
      numerical_cols.push_back(
          data::NumericalColumn(name, numerical->range.first,
                                numerical->range.second, numerical->numBins()));
    }

    if (auto sequence_type = asSequence(data_type)) {
      auto [transforms, output] = sequence(name, sequence_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    }

    if (auto date_type = asDate(data_type)) {
      auto [transforms, output] = date(name, date_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    }
  }

  if (!numerical_cols.empty() || !categorical_cols.empty()) {
    auto transform = std::make_shared<data::Tabular>(
        numerical_cols, categorical_cols, TABULAR_COLUMNS_OUTPUT,
        options.contextual_columns);

    pipeline.push_back(transform);
    output_columns.push_back(TABULAR_COLUMNS_OUTPUT);
  }

  return {pipeline, output_columns};
}

void checkKeyColumn(const std::string& key_column,
                    const ColumnDataTypes& data_types,
                    const std::string& label_column) {
  if (!data_types.count(key_column)) {
    throw std::invalid_argument("Tracking key column '" + key_column +
                                "' is not specified in data_types.");
  }

  if (!asCategorical(data_types.at(key_column))) {
    throw std::invalid_argument("Tracking key column must be categorical.");
  }

  if (key_column == label_column) {
    throw std::invalid_argument(
        "Tracking key column cannot be the label column.");
  }
}

void checkTemporalConfig(const TemporalConfig& temporal_config,
                         const ColumnDataTypes& data_types) {
  if (!data_types.count(temporal_config.columnName())) {
    throw std::invalid_argument("Tracked column '" +
                                temporal_config.columnName() +
                                "' is not specified in data_types.");
  }
}

TransformSeries categoricalTemporal(
    const ColumnDataTypes& data_types, const std::string& key_column,
    const std::string& timestamp_col,
    const TemporalCategoricalConfig& categorical_temporal,
    const std::string& label_column, bool should_update_history,
    int64_t time_lag, uint32_t temporal_id) {
  std::vector<data::TransformationPtr> transformations;

  if (!asCategorical(data_types.at(categorical_temporal.column_name))) {
    throw std::invalid_argument("Expected the tracked column '" +
                                categorical_temporal.column_name +
                                "' to be categorical.");
  }

  auto tracked_column =
      asCategorical(data_types.at(categorical_temporal.column_name));

  // This is just an additional check to ensure that we don't leak labels if
  // the tracked column is the labels.
  bool tracking_labels = categorical_temporal.column_name == label_column;
  if (categorical_temporal.include_current_row && tracking_labels) {
    std::cerr << "Warning: Ignoring 'include_current_row' for temporal tracker "
                 "on column '"
              << categorical_temporal.column_name
              << "' since it is the target column." << std::endl;
  }
  bool include_current_row =
      categorical_temporal.include_current_row && !tracking_labels;

  std::string item_column =
      temporalItemIdsOutput(categorical_temporal.column_name);

  if (should_update_history || !tracking_labels) {
    // The values of the tracked column will only be used if either 1) we are
    // updating history or 2) if we are not tracking the label column, and thus
    // the values could be used as an input. If we are not updating history and
    // are tracking the label column, then this column will not be used (or
    // likely will not even be present since it's the target column).
    auto item_hash = std::make_shared<data::StringHash>(
        categorical_temporal.column_name, item_column, std::nullopt,
        tracked_column->delimiter);
    transformations.push_back(item_hash);
  }

  std::string output = temporalTrackingOutput(temporal_id);

  auto transformation = std::make_shared<data::CategoricalTemporal>(
      /* user_column= */ key_column,
      /* item_column= */ item_column,
      /* timestamp_column= */ timestamp_col,
      /* output_column= */ output,
      /* tracker_key= */ output,
      /* track_last_n= */ categorical_temporal.track_last_n,
      /* should_update_history= */ should_update_history,
      /* include_current_row= */ include_current_row,
      /* time_lag= */ time_lag);

  transformations.push_back(transformation);

  return {transformations, output};
}

TransformSeries numericalTemporal(
    const ColumnDataTypes& data_types, const std::string& key_column,
    const std::string& timestamp_col,
    const TemporalNumericalConfig& numerical_temporal,
    const std::string& label_column, bool should_update_history,
    int64_t interval_len, int64_t interval_lag, uint32_t temporal_id) {
  std::vector<data::TransformationPtr> transformations;

  if (!asNumerical(data_types.at(numerical_temporal.column_name))) {
    throw std::invalid_argument("Expected the tracked column '" +
                                numerical_temporal.column_name +
                                "' to be numerical.");
  }

  auto tracked_column =
      asNumerical(data_types.at(numerical_temporal.column_name));

  // This is just an additional check to ensure that we don't leak labels if
  // the tracked column is the labels.
  bool tracking_labels = numerical_temporal.column_name == label_column;
  if (numerical_temporal.include_current_row && tracking_labels) {
    std::cerr << "Warning: Ignoring 'include_current_row' for temporal tracker "
                 "on column '"
              << numerical_temporal.column_name
              << "' since it is the target column." << std::endl;
  }
  bool include_current_row =
      numerical_temporal.include_current_row && !tracking_labels;

  std::string value_column =
      temporalNumericalValueOutput(numerical_temporal.column_name);

  if (should_update_history || !tracking_labels) {
    auto parse_value = std::make_shared<data::CastToValue<float>>(
        numerical_temporal.column_name, value_column);
    transformations.push_back(parse_value);
  }

  std::string output = temporalTrackingOutput(temporal_id);

  auto transformation = std::make_shared<data::NumericalTemporal>(
      /* user_column= */ key_column,
      /* value_column= */ value_column,
      /* timestamp_column= */ timestamp_col,
      /* output_column= */ output,
      /* tracker_key= */ output,
      /* history_len= */ numerical_temporal.history_length,
      /* interval_len= */ interval_len,
      /* should_update_history= */ should_update_history,
      /* include_current_row= */ include_current_row,
      /* interval_lag= */ interval_lag);

  transformations.push_back(transformation);

  return {transformations, output};
}

MergedTransformSeries temporalTransformations(
    const ColumnDataTypes& data_types, const std::string& label_column,
    const TemporalRelationships& temporal_relationships,
    const TabularOptions& options, bool should_update_history) {
  if (temporal_relationships.empty()) {
    return {{}, {}};
  }

  uint32_t temporal_id = 0;

  auto [timestamp_cast, timestamp_col] = timestamp(data_types);

  std::vector<data::TransformationPtr> transformations = {timestamp_cast};
  std::vector<std::string> output_columns;

  for (const auto& [key_column, relationships] : temporal_relationships) {
    checkKeyColumn(key_column, data_types, label_column);

    for (const auto& temporal_config : relationships) {
      checkTemporalConfig(temporal_config, data_types);

      if (temporal_config.isCategorical()) {
        auto [tracker_transformations, output] = categoricalTemporal(
            /*data_types=*/data_types, /*key_column=*/key_column,
            /*timestamp_col=*/timestamp_col,
            /*categorical_temporal=*/temporal_config.asCategorical(),
            /*label_column=*/label_column,
            /*should_update_history=*/should_update_history,
            /*time_lag=*/options.timeLag(), /*temporal_id=*/temporal_id++);

        transformations.insert(transformations.end(),
                               tracker_transformations.begin(),
                               tracker_transformations.end());
        output_columns.push_back(output);
      } else if (temporal_config.isNumerical()) {
        auto [tracker_transformations, output] = numericalTemporal(
            /*data_types=*/data_types, /*key_column=*/key_column,
            /*timestamp_col=*/timestamp_col,
            /*numerical_temporal=*/temporal_config.asNumerical(),
            /*label_column=*/label_column,
            /*should_update_history=*/should_update_history,
            /*interval_len=*/options.granularity(),
            /*interval_lag=*/options.lookahead, /*temporal_id=*/temporal_id++);

        transformations.insert(transformations.end(),
                               tracker_transformations.begin(),
                               tracker_transformations.end());
        output_columns.push_back(output);
      }
    }
  }

  return {transformations, output_columns};
}

ColumnDataTypes removeTrackedColumns(
    ColumnDataTypes data_types,
    const TemporalRelationships& temporal_relationships) {
  for (const auto& [_, relationships] : temporal_relationships) {
    for (const auto& config : relationships) {
      if (data_types.count(config.columnName())) {
        data_types.erase(config.columnName());
      }
    }
  }

  return data_types;
}

std::pair<data::TransformationPtr, data::OutputColumnsList>
inputTransformations(const ColumnDataTypes& data_types,
                     const std::string& label_column,
                     const TemporalRelationships& temporal_relationships,
                     const TabularOptions& options,
                     bool should_update_history) {
  if (!data_types.count(label_column)) {
    throw std::invalid_argument(
        "Target column was not specified in data_types.");
  }

  checkNoReservedColumnNames(data_types);

  auto non_temporal_input_data_types =
      removeTrackedColumns(data_types, temporal_relationships);
  if (non_temporal_input_data_types.count(label_column)) {
    non_temporal_input_data_types.erase(label_column);
  }

  if (non_temporal_input_data_types.size() == 1 &&
      temporal_relationships.empty()) {
    // If we only have a single text input then we can skip additional feature
    // hashing and just have a single text transformation.
    auto [name, type] = *non_temporal_input_data_types.begin();

    if (auto text_type = asText(type)) {
      auto transform = std::make_shared<data::TextTokenizer>(
          name, FEATURIZED_INDICES, FEATURIZED_VALUES, text_type->tokenizer,
          text_type->encoder, text_type->lowercase, options.feature_hash_range);

      return {transform,
              {data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)}};
    }
  }

  auto [transformations, output_columns] =
      nonTemporalTransformations(non_temporal_input_data_types, options);

  auto [temporal_transformations, temporal_outputs] =
      temporalTransformations(data_types, label_column, temporal_relationships,
                              options, should_update_history);

  transformations.insert(transformations.end(),
                         temporal_transformations.begin(),
                         temporal_transformations.end());

  output_columns.insert(output_columns.end(), temporal_outputs.begin(),
                        temporal_outputs.end());

  auto feature_hash = std::make_shared<data::FeatureHash>(
      /* input_columns= */ output_columns,
      /* output_indices_column= */ FEATURIZED_INDICES,
      /* output_values_column= */ FEATURIZED_VALUES,
      /* hash_range= */ options.feature_hash_range);

  transformations.push_back(feature_hash);

  auto pipeline = data::Pipeline::make(transformations);

  return {pipeline,
          {data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)}};
}

MergedTransformSeries nonTemporalTransformations(
    ColumnDataTypes data_types, const std::string& label_column,
    const TabularOptions& options) {
  if (!data_types.count(label_column)) {
    throw std::invalid_argument(
        "Target column was not specified in data_types.");
  }

  checkNoReservedColumnNames(data_types);

  data_types.erase(label_column);

  return nonTemporalTransformations(data_types, options);
}

}  // namespace thirdai::automl