#include "Featurizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <memory>
#include <stdexcept>

namespace thirdai::automl {

Featurizer::Featurizer(
    data::ColumnDataTypes data_types,
    const data::TemporalRelationships& temporal_relationships,
    const std::string& label_column,
    thirdai::data::TransformationPtr label_transform,
    thirdai::data::OutputColumnsList bolt_label_columns,
    const data::TabularOptions& options)
    : _label_transform(std::move(label_transform)),
      _bolt_label_columns(std::move(bolt_label_columns)),
      _delimiter(options.delimiter),
      _state(std::make_shared<thirdai::data::State>()) {
  std::tie(_input_transform, _bolt_input_columns) =
      inputTransformations(data_types, label_column, temporal_relationships,
                           options, /* should_update_history= */ true);

  _input_transform_non_updating =
      inputTransformations(data_types, label_column, temporal_relationships,
                           options, /* should_update_history= */ false)
          .first;

  if (data_types.size() == 2 && temporal_relationships.empty()) {
    auto cat_label = data::asCategorical(data_types.at(label_column));
    data_types.erase(label_column);
    auto text_type = data::asText(data_types.begin()->second);
    if (text_type && cat_label) {
      _text_dataset = TextDatasetConfig(data_types.begin()->first, label_column,
                                        cat_label->delimiter);
    }
  }
}

thirdai::data::LoaderPtr Featurizer::getDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  return getDataLoaderHelper(data_source, batch_size, shuffle, verbose,
                             shuffle_config);
}

thirdai::data::LoaderPtr Featurizer::getColdStartDataLoader(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool fast_approximation,
    size_t batch_size, bool shuffle, bool verbose,
    dataset::DatasetShuffleConfig shuffle_config) {
  auto cold_start = coldStartTransform(strong_column_names, weak_column_names,
                                       fast_approximation);

  return getDataLoaderHelper(data_source, batch_size, shuffle, verbose,
                             shuffle_config, cold_start);
}

thirdai::data::LoaderPtr Featurizer::getDataLoaderHelper(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config,
    const thirdai::data::TransformationPtr& cold_start_transform) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  auto data_iter =
      thirdai::data::CsvIterator::make(csv_data_source, _delimiter);

  std::vector<thirdai::data::TransformationPtr> transformations;
  if (cold_start_transform) {
    transformations.push_back(cold_start_transform);
  }
  transformations.push_back(_input_transform);
  transformations.push_back(_label_transform);

  auto transformation_list =
      thirdai::data::TransformationList::make(transformations);

  return thirdai::data::Loader::make(
      data_iter, transformation_list, _state, _bolt_input_columns,
      _bolt_label_columns, /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_buffer_size= */ shuffle_config.min_buffer_size,
      /* shuffle_seed= */ shuffle_config.seed);
}

bolt::TensorList Featurizer::featurizeInput(const MapInput& sample) {
  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns = _input_transform_non_updating->apply(std::move(columns), *_state);

  return thirdai::data::toTensors(columns, _bolt_input_columns);
}

bolt::TensorList Featurizer::featurizeInputBatch(const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform_non_updating->apply(std::move(columns), *_state);

  return thirdai::data::toTensors(columns, _bolt_input_columns);
}

bolt::TensorList Featurizer::featurizeInputColdStart(
    MapInput sample, const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  auto cold_start = coldStartTransform(strong_column_names, weak_column_names);
  // Currently the cold start transformation expects a label column that it
  // repeats when it maps a single set of phrases to multiple samples. In the
  // future it should be extended to not require this, and just duplicate any
  // other columns it finds.
  sample[_text_dataset->labelColumn()] = "";

  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns = cold_start->apply(columns, *_state);
  columns = _input_transform_non_updating->apply(columns, *_state);

  return thirdai::data::toTensors(columns, _bolt_input_columns);
}

std::pair<bolt::TensorList, bolt::TensorList>
Featurizer::featurizeTrainingBatch(const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform->apply(columns, *_state);
  columns = _label_transform->apply(columns, *_state);

  auto data = thirdai::data::toTensors(columns, _bolt_input_columns);

  auto labels = thirdai::data::toTensors(columns, _bolt_label_columns);

  return std::make_pair(std::move(data), std::move(labels));
}

thirdai::data::TransformationPtr Featurizer::coldStartTransform(
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    bool fast_approximation) {
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

auto asTransformationList(const thirdai::data::TransformationPtr& t) {
  return std::dynamic_pointer_cast<thirdai::data::TransformationList>(t);
}

auto asTemporal(const thirdai::data::TransformationPtr& t) {
  return std::dynamic_pointer_cast<thirdai::data::CategoricalTemporal>(t);
}

bool hasTemporalTransformation(const thirdai::data::TransformationPtr& t) {
  std::queue<thirdai::data::TransformationPtr> queue;
  queue.push(t);

  while (!queue.empty()) {
    auto next = queue.front();
    queue.pop();
    if (asTemporal(next)) {
      return true;
    }
    if (auto list = asTransformationList(next)) {
      for (const auto& transform : list->transformations()) {
        queue.push(transform);
      }
    }
  }

  return false;
}

bool Featurizer::hasTemporalTransformations() const {
  return hasTemporalTransformation(_input_transform);
}

void Featurizer::updateTemporalTrackers(const MapInput& sample) {
  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);
  _input_transform->apply(columns, *_state);
}

void Featurizer::updateTemporalTrackersBatch(const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);
  _input_transform->apply(columns, *_state);
}

void Featurizer::resetTemporalTrackers() { _state->clearHistoryTrackers(); }

template void Featurizer::serialize(cereal::BinaryInputArchive&);
template void Featurizer::serialize(cereal::BinaryOutputArchive&);

template <typename Archive>
void Featurizer::serialize(Archive& archive) {
  archive(_input_transform, _input_transform_non_updating, _label_transform,
          _bolt_input_columns, _bolt_label_columns, _delimiter, _state,
          _text_dataset);
}

}  // namespace thirdai::automl