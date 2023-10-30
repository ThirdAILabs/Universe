#include "MachFeaturizer.h"

namespace thirdai::automl {

template void MachFeaturizer::serialize(cereal::BinaryInputArchive&);
template void MachFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachFeaturizer::serialize(Archive& archive) {
  archive(_delimiter, _label_delimiter, _state, _tracking_input_transformation,
          _input_indices_column, _input_values_column,
          _const_input_transformation, _label_transformation,
          _string_to_int_buckets, _text_dataset_config);
}

MachFeaturizer::MachFeaturizer(
    ColumnDataTypes data_types, const CategoricalDataTypePtr& target_config,
    const TemporalRelationships& temporal_relationships,
    const std::string& label_column, const TabularOptions& options)
    : _delimiter(options.delimiter),
      _label_delimiter(target_config->delimiter),
      _state(data::State::make()) {
  data::OutputColumnsList input_columns;
  std::tie(_tracking_input_transformation, input_columns) =
      inputTransformations(data_types, label_column, temporal_relationships,
                           options, /* should_update_history= */ true);

  _input_indices_column = input_columns.front().indices();
  _input_values_column = input_columns.front().values().value();

  _const_input_transformation =
      inputTransformations(data_types, label_column, temporal_relationships,
                           options, /* should_update_history= */ false)
          .first;

  if (target_config->delimiter) {
    _label_transformation = std::make_shared<data::StringToTokenArray>(
        label_column, modelLabelColumn(), target_config->delimiter.value(),
        std::numeric_limits<uint32_t>::max());
  } else {
    _label_transformation = std::make_shared<data::StringToToken>(
        label_column, modelLabelColumn(), std::numeric_limits<uint32_t>::max());
  }

  if (data_types.size() == 2 && temporal_relationships.empty()) {
    auto cat_label = asCategorical(data_types.at(label_column));
    data_types.erase(label_column);
    auto text_type = asText(data_types.begin()->second);
    if (text_type && cat_label) {
      _text_dataset_config = TextDatasetConfig(
          data_types.begin()->first, label_column, cat_label->delimiter);
    }
  }
}
data::ColumnMap MachFeaturizer::addLabelColumn(data::ColumnMap&& columns,
                                               uint32_t label) const {
  data::ColumnPtr label_column;
  if (_label_delimiter) {
    std::vector<std::vector<uint32_t>> label_column_data(columns.numRows(),
                                                         {label});
    label_column = data::ArrayColumn<uint32_t>::make(
        std::move(label_column_data), std::numeric_limits<uint32_t>::max());
  } else {
    std::vector<uint32_t> label_column_data(columns.numRows(), label);
    label_column = data::ValueColumn<uint32_t>::make(
        std::move(label_column_data), std::numeric_limits<uint32_t>::max());
  }
  columns.setColumn(modelLabelColumn(), label_column);
  return columns;
}
std::pair<data::ColumnMap, data::ColumnMap>
MachFeaturizer::associationColumnMaps(
    const std::vector<std::pair<std::string, std::string>>& samples) const {
  std::vector<std::string> from_texts(samples.size());
  std::vector<std::string> to_texts(samples.size());
  for (uint32_t i = 0; i < samples.size(); i++) {
    from_texts[i] = samples[i].first;
    to_texts[i] = samples[i].second;
  }

  data::ColumnMap from_columns(
      {{textDatasetConfig().textColumn(),
        data::ValueColumn<std::string>::make(std::move(from_texts))}});
  data::ColumnMap to_columns(
      {{textDatasetConfig().textColumn(),
        data::ValueColumn<std::string>::make(std::move(to_texts))}});
  return {std::move(from_columns), std::move(to_columns)};
}
data::ColumnMap MachFeaturizer::upvoteLabeledColumnMap(
    const std::vector<std::pair<std::string, uint32_t>>& samples) const {
  assertTextModel();

  std::vector<std::string> from_texts(samples.size());
  std::vector<uint32_t> to_labels(samples.size());
  for (uint32_t i = 0; i < samples.size(); i++) {
    from_texts[i] = samples[i].first;
    to_labels[i] = samples[i].second;
  }

  return data::ColumnMap(
      {{textDatasetConfig().textColumn(),
        data::ValueColumn<std::string>::make(std::move(from_texts))},
       {modelLabelColumn(),
        data::ValueColumn<uint32_t>::make(
            std::move(to_labels), std::numeric_limits<uint32_t>::max())}});
}
data::TransformationPtr MachFeaturizer::coldstartTransformation(
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    bool fast_approximation) const {
  assertTextModel();

  if (fast_approximation) {
    std::vector<std::string> all_columns = weak_column_names;
    all_columns.insert(all_columns.end(), strong_column_names.begin(),
                       strong_column_names.end());
    return std::make_shared<data::StringConcat>(
        all_columns, textDatasetConfig().textColumn());
  }

  return std::make_shared<data::ColdStartTextAugmentation>(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ textDatasetConfig().labelColumn(),
      /* output_column_name= */ textDatasetConfig().textColumn());
}
bool MachFeaturizer::hasTemporalTransformations() const {
  std::queue<data::TransformationPtr> queue;
  queue.push(_tracking_input_transformation);

  while (!queue.empty()) {
    auto next = queue.front();
    queue.pop();
    if (std::dynamic_pointer_cast<data::CategoricalTemporal>(next)) {
      return true;
    }
    if (auto list = std::dynamic_pointer_cast<data::Pipeline>(next)) {
      for (const auto& transform : list->transformations()) {
        queue.push(transform);
      }
    }
  }

  return false;
}
}  // namespace thirdai::automl