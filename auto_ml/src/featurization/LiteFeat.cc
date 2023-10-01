#include "LiteFeat.h"
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <optional>

namespace thirdai::automl {

thirdai::data::TransformationPtr labelTransform(
    const std::string& target_name,
    const data::CategoricalDataType& target_config) {
  if (!target_config.delimiter) {
    return std::make_shared<thirdai::data::StringToToken>(
        target_name, FEATURIZED_LABELS, /* dim= */ std::nullopt);
  }
  return std::make_shared<thirdai::data::StringToTokenArray>(
      target_name, FEATURIZED_LABELS, target_config.delimiter.value(),
      /* dim= */ std::nullopt);
}

LiteFeat::LiteFeat(
    data::ColumnDataTypes data_types,
    const data::UserProvidedTemporalRelationships& user_temporal_relationships,
    const std::string& label_column,
    thirdai::data::ValueFillType label_value_fill,
    const data::TabularOptions& options)
    : _has_temporal_transform(!user_temporal_relationships.empty()) {
  auto temporal_relationships = data::TemporalRelationshipsAutotuner::autotune(
      data_types, user_temporal_relationships, options.lookahead);

  auto [train_input_transform, bolt_input_columns] =
      inputTransformations(data_types, label_column, temporal_relationships,
                           options, /* should_update_history= */ true);

  auto label_transform = labelTransform(
      label_column, *data::asCategorical(data_types.at(label_column)));

  auto train_transform = thirdai::data::TransformationList::make(
      {train_input_transform, label_transform});
  auto bolt_label_columns =
      thirdai::data::OutputColumns(FEATURIZED_LABELS, label_value_fill);

  _train_transform = TransformConfig(
      /* _transform= */ std::move(train_transform),
      /* _input_columns= */ bolt_input_columns,
      /* _label_columns= */ {{bolt_label_columns}});

  auto [infer_input_transform, _] =
      inputTransformations(data_types, label_column, temporal_relationships,
                           options, /* should_update_history= */ true);

  _infer_transform = TransformConfig(
      /* _transform= */ std::move(infer_input_transform),
      /* _input_columns= */ bolt_input_columns,
      /* _label_columns= */ std::nullopt);

  auto intro_transform = thirdai::data::TransformationList::make(
      {infer_input_transform, label_transform});

  _intro_transform = TransformConfig(
      /* _transform= */ std::move(intro_transform),
      /* _input_columns= */ bolt_input_columns,
      /* _label_columns= */ std::nullopt);

  _text_dataset = TextDatasetConfig::fromDataTypes(
      std::move(data_types), temporal_relationships, label_column);
}

thirdai::data::TransformationPtr LiteFeat::unsupAugmenter(
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    bool fast_approximation) const {
  if (!_text_dataset) {
    throw std::invalid_argument("Cold start is not supported for this model.");
  }

  if (fast_approximation) {
    std::vector<std::string> all_columns = weak_column_names;
    all_columns.insert(all_columns.end(), strong_column_names.begin(),
                       strong_column_names.end());
    return std::make_shared<thirdai::data::StringConcat>(
        all_columns, _text_dataset->textColumn());
  }

  return std::make_shared<thirdai::data::ColdStartTextAugmentation>(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ _text_dataset->labelColumn(),
      /* output_column_name= */ _text_dataset->textColumn());
}

}  // namespace thirdai::automl