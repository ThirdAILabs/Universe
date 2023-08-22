#include "TabularTransformations.h"
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl {

// This represents the transformations and the output for a column in the input.
using TransformSeries =
    std::pair<std::vector<thirdai::data::TransformationPtr>, std::string>;

// This represents the transformations and outputs for a set of columns in the
// input.
using MergedTransformSeries =
    std::pair<std::vector<thirdai::data::TransformationPtr>,
              std::vector<std::string>>;

TransformSeries text(const std::string& column_name,
                     const data::TextDataTypePtr& text,
                     size_t dim = std::numeric_limits<uint32_t>::max()) {
  std::string output = textOutputColumn(column_name);

  auto transformation = std::make_shared<thirdai::data::TextTokenizer>(
      /* input_column= */ column_name, /* output_column= */ output,
      /* tokenizer= */ text->tokenizer, /* encoder= */ text->encoder,
      /* lowercase= */ text->lowercase, /* dim= */ dim);

  return {{transformation}, output};
}

TransformSeries categorical(const std::string& column_name,
                            const data::CategoricalDataTypePtr& categorical) {
  std::string output = categoricalOutputColumn(column_name);

  if (categorical->delimiter) {
    auto tok = dataset::NaiveSplitTokenizer::make(*categorical->delimiter);
    auto enc = dataset::NGramEncoder::make(/* n = */ 1);

    auto transformation = std::make_shared<thirdai::data::TextTokenizer>(
        /* input_column= */ column_name, /* output_column= */ output,
        /* tokenizer= */ tok, /* encoder= */ enc, /* lowercase= */ false,
        /* dim= */ std::numeric_limits<uint32_t>::max());

    return {{transformation}, output};
  }

  auto transformation = std::make_shared<thirdai::data::StringHash>(
      /* input_column_name= */ column_name, /* output_column_name= */ output);

  return {{transformation}, output};
}

TransformSeries binning(const std::string& column_name,
                        const data::NumericalDataTypePtr& numerical) {
  std::string output = binningOutputColumn(column_name);

  auto cast = std::make_shared<thirdai::data::StringToDecimal>(column_name,
                                                               column_name);

  auto transformation = std::make_shared<thirdai::data::BinningTransformation>(
      /* input_column_name= */ column_name,
      /* output_column_name= */ output,
      /* inclusive_min_value= */ numerical->range.first,
      /* exlusive_max_value= */ numerical->range.second,
      /* num_bins= */ numerical->numBins());

  return {{cast, transformation}, output};
}

TransformSeries sequence(const std::string& column_name,
                         const data::SequenceDataTypePtr& sequence) {
  std::string output = sequenceOutputColumn(column_name);

  auto hash = std::make_shared<thirdai::data::StringHash>(
      column_name, column_name, sequence->delimiter);

  auto transformation = std::make_shared<thirdai::data::HashPositionTransform>(
      column_name, output,
      /* hash_range= */ std::numeric_limits<uint32_t>::max());

  return {{hash, transformation}, output};
}

TransformSeries date(const std::string& column_name,
                     const data::DateDataTypePtr& date) {
  (void)date;

  std::string output = dateOutputColumn(column_name);

  auto transformation = std::make_shared<thirdai::data::Date>(
      /* input_column_name= */ column_name, /* output_column_name= */ output);

  return {{transformation}, output};
}

TransformSeries crossColumnPaigrams(
    const std::vector<std::string>& tabular_columns) {
  auto transformation = std::make_shared<thirdai::data::CrossColumnPairgrams>(
      /* input_column_names= */ tabular_columns,
      /* output_column_name= */ CROSS_COLUMN_PAIRGRAMS_OUTPUT,
      /* hash_range= */ std::numeric_limits<uint32_t>::max());

  return {{transformation}, CROSS_COLUMN_PAIRGRAMS_OUTPUT};
}

TransformSeries timestamp(const data::ColumnDataTypes& data_types) {
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

  auto transformation = std::make_shared<thirdai::data::StringToTimestamp>(
      /* input_column_name= */ *timestamp_column,
      /* output_column_name= */ TIMESTAMP_OUTPUT, /* format= */ "%Y-%m-%d");

  return {{transformation}, TIMESTAMP_OUTPUT};
}

MergedTransformSeries nonTemporalTransformations(
    const data::ColumnDataTypes& data_types,
    const data::TabularOptions& options) {
  std::vector<thirdai::data::TransformationPtr> pipeline;
  std::vector<std::string> output_columns;
  std::vector<std::string> tabular_columns;

  for (const auto& [name, data_type] : data_types) {
    if (auto text_type = data::asText(data_type)) {
      auto [transforms, output] = text(name, text_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    }

    if (auto cat_type = data::asCategorical(data_type)) {
      auto [transforms, output] = categorical(name, cat_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      if (!cat_type->delimiter) {
        tabular_columns.push_back(output);
      } else {
        output_columns.push_back(output);
      }
    }

    if (auto numerical = data::asNumerical(data_type)) {
      auto [transforms, output] = binning(name, numerical);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      tabular_columns.push_back(output);
    }

    if (auto sequence_type = data::asSequence(data_type)) {
      auto [transforms, output] = sequence(name, sequence_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    }

    if (auto date_type = data::asDate(data_type)) {
      auto [transforms, output] = date(name, date_type);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    }
  }

  if (!tabular_columns.empty()) {
    if (options.contextual_columns) {
      auto [transforms, output] = crossColumnPaigrams(tabular_columns);
      pipeline.insert(pipeline.end(), transforms.begin(), transforms.end());
      output_columns.push_back(output);
    } else {
      output_columns.insert(output_columns.end(), tabular_columns.begin(),
                            tabular_columns.end());
    }
  }

  return {pipeline, output_columns};
}

void checkKeyColumn(const std::string& key_column,
                    const data::ColumnDataTypes& data_types,
                    const std::string& label_column) {
  if (!data_types.count(key_column)) {
    throw std::invalid_argument("Tracking key column '" + key_column +
                                "' is not specified in data_types.");
  }

  if (!data::asCategorical(data_types.at(key_column))) {
    throw std::invalid_argument("Tracking key column must be categorical.");
  }

  if (key_column == label_column) {
    throw std::invalid_argument(
        "Tracking key column cannot be the label column.");
  }
}

void checkTemporalConfig(const data::TemporalConfig& temporal_config,
                         const data::ColumnDataTypes& data_types) {
  if (!temporal_config.isCategorical()) {
    throw std::invalid_argument(
        "Only categorical temporal tracking is supported.");
  }

  auto categorical_temporal = temporal_config.asCategorical();

  if (!data_types.count(categorical_temporal.column_name)) {
    throw std::invalid_argument("Tracked column '" +
                                categorical_temporal.column_name +
                                "' is not specified in data_types.");
  }

  if (!data::asCategorical(data_types.at(categorical_temporal.column_name))) {
    throw std::invalid_argument("Expected the tracked column '" +
                                categorical_temporal.column_name +
                                "' to be categorical.");
  }
}

MergedTransformSeries temporalTransformations(
    const data::ColumnDataTypes& data_types, const std::string& label_column,
    const data::TemporalRelationships& temporal_relationships,
    const data::TabularOptions& options, bool should_update_history) {
  if (temporal_relationships.empty()) {
    return {{}, {}};
  }

  uint32_t temporal_id = 0;

  auto [timestamp_cast, timestamp_col] = timestamp(data_types);

  std::vector<thirdai::data::TransformationPtr> transformations = {
      timestamp_cast};
  std::vector<std::string> output_columns;

  for (const auto& [key_column, relationships] : temporal_relationships) {
    checkKeyColumn(key_column, data_types, label_column);

    for (const auto& temporal_config : relationships) {
      checkTemporalConfig(temporal_config, data_types);

      auto categorical_temporal = temporal_config.asCategorical();

      auto tracked_column =
          data::asCategorical(data_types.at(categorical_temporal.column_name));

      // This is just an additional check to ensure that we don't leak labels if
      // the tracked column is the labels.
      bool include_current_row =
          categorical_temporal.include_current_row &&
          (categorical_temporal.column_name != label_column);

      std::string item_column =
          temporalItemIdsOutput(categorical_temporal.column_name);

      auto item_hash = std::make_shared<thirdai::data::StringHash>(
          categorical_temporal.column_name, item_column,
          tracked_column->delimiter);
      transformations.push_back(item_hash);

      std::string output = temporalTrackingOutput(temporal_id++);

      auto transformation =
          std::make_shared<thirdai::data::CategoricalTemporal>(
              /* user_column= */ key_column,
              /* item_column= */ item_column,
              /* timestamp_column= */ timestamp_col,
              /* output_column= */ output,
              /* tracker_key= */ output,
              /* track_last_n= */ categorical_temporal.track_last_n,
              /* should_update_history= */ should_update_history,
              /* include_current_row= */ include_current_row,
              /* time_lag= */ options.timeLag());

      transformations.push_back(transformation);
      output_columns.push_back(output);
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

std::pair<thirdai::data::TransformationPtr, thirdai::data::OutputColumnsList>
inputTransformations(const data::ColumnDataTypes& data_types,
                     const std::string& label_column,
                     const data::TemporalRelationships& temporal_relationships,
                     const data::TabularOptions& options,
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

    if (auto text_type = data::asText(type)) {
      auto [transforms, output_name] =
          text(name, text_type, /* dim= */ options.feature_hash_range);

      return {transforms.at(0), {thirdai::data::OutputColumns(output_name)}};
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

  auto feature_hash = std::make_shared<thirdai::data::FeatureHash>(
      /* input_columns= */ output_columns,
      /* output_indices_column= */ FEATURE_HASH_INDICES,
      /* output_values_column= */ FEATURE_HASH_VALUES,
      /* hash_range= */ options.feature_hash_range);

  transformations.push_back(feature_hash);

  auto t_list =
      std::make_shared<thirdai::data::TransformationList>(transformations);

  return {t_list,
          {thirdai::data::OutputColumns(FEATURE_HASH_INDICES,
                                        FEATURE_HASH_VALUES)}};
}

}  // namespace thirdai::automl