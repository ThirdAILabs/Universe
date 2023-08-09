#include "TabularTransformations.h"
#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
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

using UniqueColumnNamer = std::function<std::string(std::string)>;

CreatedTransformation textT(const std::string& column_name,
                            const data::TextDataTypePtr& text,
                            const UniqueColumnNamer& column_namer,
                            size_t dim = std::numeric_limits<uint32_t>::max()) {
  std::string output = column_namer("__" + column_name + "_tokenized__");

  auto transformation = std::make_shared<thirdai::data::TextTokenizer>(
      /* input_column= */ column_name, /* output_column= */ output,
      /* tokenizer= */ text->tokenizer, /* encoder= */ text->encoder,
      /* lowercase= */ text->lowercase, /* dim= */ dim);

  return {transformation, output};
}

CreatedTransformation categoricalT(
    const std::string& column_name,
    const data::CategoricalDataTypePtr& categorical,
    const UniqueColumnNamer& column_namer) {
  std::string output = column_namer("__" + column_name + "_categorical__");

  if (categorical->delimiter) {
    auto tok = dataset::NaiveSplitTokenizer::make(*categorical->delimiter);
    auto enc = dataset::NGramEncoder::make(/* n = */ 1);

    auto transformation = std::make_shared<thirdai::data::TextTokenizer>(
        /* input_column= */ column_name, /* output_column= */ output,
        /* tokenizer= */ tok, /* encoder= */ enc, /* lowercase= */ false,
        /* dim= */ std::numeric_limits<uint32_t>::max());

    return {transformation, output};
  }

  auto transformation = std::make_shared<thirdai::data::StringHash>(
      /* input_column_name= */ column_name, /* output_column_name= */ output);

  return {transformation, output};
}

CreatedTransformation binningT(const std::string& column_name,
                               const data::NumericalDataTypePtr& numerical,
                               const UniqueColumnNamer& column_namer) {
  std::string output = column_namer("__" + column_name + "_binned__");

  auto transformation = std::make_shared<thirdai::data::BinningTransformation>(
      /* input_column_name= */ column_name, /* output_column_name= */ output,
      /* inclusive_min_value= */ numerical->range.first,
      /* exlusive_max_value= */ numerical->range.second,
      /* num_bins= */ numerical->numBins());

  return {transformation, output};
}

CreatedTransformation dateT(const std::string& column_name,
                            const data::DateDataTypePtr& date,
                            const UniqueColumnNamer& column_namer) {
  (void)date;

  std::string output = column_namer("__" + column_name + "_date__");

  auto transformation = std::make_shared<thirdai::data::Date>(
      /* input_column_name= */ column_name, /* output_column_name= */ output);

  return {transformation, output};
}

CreatedTransformation crossColumnPaigramsT(
    const std::vector<std::string>& tabular_columns,
    const UniqueColumnNamer& column_namer) {
  std::string output = column_namer("__contextual_columns__");

  auto transformation = std::make_shared<thirdai::data::CrossColumnPairgrams>(
      /* input_column_names= */ tabular_columns,
      /* output_column_name= */ output,
      /* hash_range= */ std::numeric_limits<uint32_t>::max());

  return {transformation, output};
}

CreatedTransformation timestampT(const data::ColumnDataTypes& data_types,
                                 const UniqueColumnNamer& column_namer) {
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

  std::string output = column_namer("__" + *timestamp_column + "_timestamp__");

  auto transformation = std::make_shared<thirdai::data::StringToTimestamp>(
      /* input_column_name= */ *timestamp_column,
      /* output_column_name= */ output, /* format= */ "%Y-%m-%d");

  return {transformation, output};
}

CreatedTransformations nonTemporalTransformations(
    const data::ColumnDataTypes& data_types,
    const data::TabularOptions& options,
    const UniqueColumnNamer& column_namer) {
  std::vector<thirdai::data::TransformationPtr> transformations;
  std::vector<std::string> output_columns;
  std::vector<std::string> tabular_columns;

  for (const auto& [name, data_type] : data_types) {
    if (auto text = data::asText(data_type)) {
      auto [transform, output] = textT(name, text, column_namer);
      transformations.push_back(transform);
      output_columns.push_back(output);
    }

    if (auto categorical = data::asCategorical(data_type)) {
      auto [transform, output] = categoricalT(name, categorical, column_namer);
      transformations.push_back(transform);
      if (!categorical->delimiter) {
        tabular_columns.push_back(output);
      } else {
        output_columns.push_back(output);
      }
    }

    if (auto numerical = data::asNumerical(data_type)) {
      auto [transform, output] = binningT(name, numerical, column_namer);
      transformations.push_back(transform);
      tabular_columns.push_back(output);
    }

    if (auto sequence = data::asSequence(data_type)) {
      throw std::runtime_error("TODO(Nicholas, Geordie): sequence data type.");
    }

    if (auto date = data::asDate(data_type)) {
      auto [transform, output] = dateT(name, date, column_namer);
      transformations.push_back(transform);
      output_columns.push_back(output);
    }
  }

  if (!tabular_columns.empty()) {
    if (options.contextual_columns) {
      auto [xcol, output] = crossColumnPaigramsT(tabular_columns, column_namer);
      transformations.push_back(xcol);
      output_columns.push_back(output);
    } else {
      output_columns.insert(output_columns.end(), tabular_columns.begin(),
                            tabular_columns.end());
    }
  }

  return {transformations, output_columns};
}

void checkKeyColumn(const std::string& key_column,
                    const data::ColumnDataTypes& data_types,
                    const std::string& label_column) {
  if (!data_types.count(key_column)) {
    throw std::invalid_argument("Tracking key column '" + key_column +
                                "' is not specified in data_types.");
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

CreatedTransformations temporalTransformations(
    const data::ColumnDataTypes& data_types, const std::string& label_column,
    const data::TemporalRelationships& temporal_relationships,
    const data::TabularOptions& options, bool should_update_history,
    const UniqueColumnNamer& column_namer) {
  if (temporal_relationships.empty()) {
    return {{}, {}};
  }

  uint32_t temporal_id = 0;

  auto [timestamp_cast, timestamp_col] = timestampT(data_types, column_namer);

  std::vector<thirdai::data::TransformationPtr> transformations = {
      timestamp_cast};
  std::vector<std::string> output_columns;

  for (const auto& [key_column, relationships] : temporal_relationships) {
    checkKeyColumn(key_column, data_types, label_column);

    for (const auto& temporal_config : relationships) {
      checkTemporalConfig(temporal_config, data_types);

      auto categorical_temporal = temporal_config.asCategorical();

      // This is just an additional check to ensure that we don't leak labels if
      // the tracked column is the labels.
      bool include_current_row =
          categorical_temporal.include_current_row &&
          (categorical_temporal.column_name != label_column);

      std::string output = column_namer("__categorical_temporal_" +
                                        std::to_string(temporal_id++) + "__");

      auto transformation =
          std::make_shared<thirdai::data::CategoricalTemporal>(
              /* user_column= */ key_column,
              /* item_column= */ categorical_temporal.column_name,
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

std::string uniqueColumnName(std::string name,
                             const data::ColumnDataTypes& data_types) {
  while (data_types.count(name)) {
    name += utils::uuid::getRandomHexString(/* num_bytes_randomness= */ 4);
  }
  return name;
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
  if (non_temporal_input_data_types.count(label_column)) {
    non_temporal_input_data_types.erase(label_column);
  }

  UniqueColumnNamer column_namer = [&data_types](const std::string& name) {
    return uniqueColumnName(name, data_types);
  };

  if (non_temporal_input_data_types.size() == 1 &&
      temporal_relationships.empty()) {
    // If we only have a single text input then we can skip additional feature
    // hashing and just have a single text transformation.
    auto text_column = *non_temporal_input_data_types.begin();

    if (auto text = data::asText(text_column.second)) {
      auto [transformation, output_name] =
          textT(text_column.first, text, column_namer,
                /* dim= */ options.feature_hash_range);

      return {transformation, {{output_name, std::nullopt}}};
    }
  }

  auto [transformations, output_columns] = nonTemporalTransformations(
      non_temporal_input_data_types, options, column_namer);

  auto [temporal_transformations, temporal_outputs] =
      temporalTransformations(data_types, label_column, temporal_relationships,
                              options, should_update_history, column_namer);

  transformations.insert(transformations.end(),
                         temporal_transformations.begin(),
                         temporal_transformations.end());

  output_columns.insert(output_columns.end(), temporal_outputs.begin(),
                        temporal_outputs.end());

  std::string output_indices = column_namer("__featurized_input_indices__");
  std::string output_values = column_namer("__featurized_input_values__");

  auto feature_hash = std::make_shared<thirdai::data::FeatureHash>(
      /* input_columns= */ output_columns,
      /* output_indices_column= */ output_indices,
      /* output_values_column= */ output_values,
      /* hash_range= */ options.feature_hash_range);

  transformations.push_back(feature_hash);

  auto t_list =
      std::make_shared<thirdai::data::TransformationList>(transformations);

  return {t_list, {{output_indices, output_values}}};
}

}  // namespace thirdai::automl