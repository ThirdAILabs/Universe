#include "TabularTransformations.h"
#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <utils/UUID.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl {

using CreatedTransformation =
    std::pair<thirdai::data::TransformationPtr, std::string>;

using CreatedTransformations =
    std::pair<std::vector<thirdai::data::TransformationPtr>,
              std::vector<std::string>>;

std::string uniqueColumnName(std::string name,
                             const data::ColumnDataTypes& data_types) {
  while (data_types.count(name)) {
    name += utils::uuid::getRandomHexString(/* num_bytes_randomness= */ 4);
  }
  return name;
}

CreatedTransformation textTransformation(
    const data::ColumnDataTypes& data_types, const std::string& column_name,
    const data::TextDataTypePtr& text,
    size_t dim = std::numeric_limits<uint32_t>::max()) {
  std::string output_name = "__" + column_name + "_tokenized__";

  auto transformation = std::make_shared<thirdai::data::TextTokenizer>(
      column_name, uniqueColumnName(output_name, data_types), text->tokenizer,
      text->encoder, text->lowercase, dim);

  return {transformation, output_name};
}

CreatedTransformation categoricalTransformation(
    const data::ColumnDataTypes& data_types, const std::string& column_name,
    const data::CategoricalDataTypePtr& categorical) {
  std::string output_name = "__" + column_name + "_categories__";

  if (categorical->delimiter) {
    auto transformation = std::make_shared<thirdai::data::TextTokenizer>(
        column_name, uniqueColumnName(output_name, data_types),
        dataset::NaiveSplitTokenizer::make(*categorical->delimiter),
        dataset::NGramEncoder::make(/* n = */ 1), /* lower_case= */ false,
        /* dim= */ std::numeric_limits<uint32_t>::max());
    return {transformation, output_name};
  }

  auto transformation =
      std::make_shared<thirdai::data::StringHash>(column_name, output_name);

  return {transformation, output_name};
}

CreatedTransformation binningTranformation(
    const data::ColumnDataTypes& data_types, const std::string& column_name,
    const data::NumericalDataTypePtr& numerical) {
  std::string output_name = "__" + column_name + "_binned__";

  auto transformation = std::make_shared<thirdai::data::BinningTransformation>(
      column_name, uniqueColumnName(output_name, data_types),
      numerical->range.first, numerical->range.second, numerical->numBins());

  return {transformation, output_name};
}

CreatedTransformations nonTemporalTransformations(
    const data::ColumnDataTypes& data_types,
    const data::TabularOptions& options) {
  std::vector<thirdai::data::TransformationPtr> transformations;
  std::vector<std::string> output_columns;
  std::vector<std::string> tabular_columns;

  for (const auto& [name, data_type] : data_types) {
    if (auto text = data::asText(data_type)) {
      auto [transformation, output_name] =
          textTransformation(data_types, name, text);
      transformations.push_back(transformation);
      output_columns.push_back(output_name);
    }

    if (auto categorical = data::asCategorical(data_type)) {
      auto [transformation, output_name] =
          categoricalTransformation(data_types, name, categorical);
      transformations.push_back(transformation);
      if (!categorical->delimiter) {
        tabular_columns.push_back(output_name);
      } else {
        output_columns.push_back(output_name);
      }
    }

    if (auto numerical = data::asNumerical(data_type)) {
      auto [transformation, output_name] =
          binningTranformation(data_types, name, numerical);
      transformations.push_back(transformation);
      tabular_columns.push_back(output_name);
    }

    // TODO(Nicholas): Sequence, Date
  }

  if (!tabular_columns.empty()) {
    if (options.contextual_columns) {
      std::string output_name = "__contextual_columns__";
      auto transformation =
          std::make_shared<thirdai::data::CrossColumnPairgrams>(
              tabular_columns, uniqueColumnName(output_name, data_types),
              std::numeric_limits<uint32_t>::max());

      transformations.push_back(transformation);
      output_columns.push_back(output_name);
    } else {
      output_columns.insert(output_columns.end(), tabular_columns.begin(),
                            tabular_columns.end());
    }
  }

  return {transformations, output_columns};
}

CreatedTransformation timestampTransformation(
    const data::ColumnDataTypes& data_types) {
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
  // TODO(Nicholas): add str -> timestamp cast.
  return {nullptr, ""};
}

CreatedTransformations temporalTransformations(
    const data::ColumnDataTypes& data_types, const std::string& label_column,
    const data::TemporalRelationships& temporal_relationships,
    const data::TabularOptions& options, bool should_update_history) {
  uint32_t temporal_id = 0;

  auto [timestamp_cast, timestamp_column] = timestampTransformation(data_types);

  std::vector<thirdai::data::TransformationPtr> transformations = {
      timestamp_cast};
  std::vector<std::string> output_columns;

  for (const auto& [key_column, relationships] : temporal_relationships) {
    if (!data_types.count(key_column)) {
      throw std::invalid_argument("Tracking key column '" + key_column +
                                  "' is not specified in data_types.");
    }
    if (key_column == label_column) {
      throw std::invalid_argument(
          "Tracking key column cannot be the label column.");
    }
    for (const auto& temporal_config : relationships) {
      if (temporal_config.isNumerical()) {
        throw std::invalid_argument(
            "Temporal tracking on numerical columns is not supported.");
      }

      auto categorical_temporal = temporal_config.asCategorical();

      if (!data_types.count(categorical_temporal.column_name)) {
        throw std::invalid_argument("Tracked column '" +
                                    categorical_temporal.column_name +
                                    "' is not specified in data_types.");
      }

      if (!data::asCategorical(
              data_types.at(categorical_temporal.column_name))) {
        throw std::invalid_argument("Expected the tracked column '" +
                                    categorical_temporal.column_name +
                                    "' to be categorical.");
      }

      bool include_current_row =
          categorical_temporal.include_current_row &&
          (categorical_temporal.column_name != label_column);

      std::string output_name =
          "__categorical_temporal_" + std::to_string(temporal_id++) + "__";
      auto transformation =
          std::make_shared<thirdai::data::CategoricalTemporal>(
              key_column, categorical_temporal.column_name, timestamp_column,
              output_name, categorical_temporal.track_last_n,
              should_update_history, include_current_row, options.timeLag());

      transformations.push_back(transformation);
      output_columns.push_back(output_name);
    }
  }

  return {transformations, output_columns};
}

data::ColumnDataTypes removeTrackedColumns(
    data::ColumnDataTypes data_types,
    const data::TemporalRelationships& temporal_relationships) {
  for (const auto& [_, relationships] : temporal_relationships) {
    for (const auto& config : relationships) {
      if (data_types.count(config.columnName())) {
        data_types.erase(config.columnName());
      }
    }
  }

  return data_types;
}

std::pair<thirdai::data::TransformationPtr, thirdai::data::IndexValueColumnList>
inputTransformations(const data::ColumnDataTypes& data_types,
                     const std::string& label_column,
                     const data::TemporalRelationships& temporal_relationships,
                     const data::TabularOptions& options,
                     bool should_update_history) {
  if (!data_types.count(label_column)) {
    throw std::invalid_argument(
        "Target column was not specified in data_types.");
  }

  auto non_temporal_input_data_types =
      removeTrackedColumns(data_types, temporal_relationships);
  non_temporal_input_data_types.erase(label_column);

  if (non_temporal_input_data_types.size() == 1 &&
      temporal_relationships.empty()) {
    // If we only have a single text input then we can skip additional feature
    // hashing and just have a single text transformation.
    if (auto text = data::asText(data_types.begin()->second)) {
      auto [transformation, output_name] =
          textTransformation(data_types, data_types.begin()->first, text,
                             /* dim= */ options.feature_hash_range);

      return {transformation, {{output_name, std::nullopt}}};
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

  std::string output_indices =
      uniqueColumnName("__featurized_input_indices__", data_types);
  std::string output_values =
      uniqueColumnName("__featurized_input_values__", data_types);

  auto feature_hash = std::make_shared<thirdai::data::FeatureHash>(
      output_columns, output_indices, output_values,
      options.feature_hash_range);

  return {feature_hash, {{output_indices, output_values}}};
}

}  // namespace thirdai::automl