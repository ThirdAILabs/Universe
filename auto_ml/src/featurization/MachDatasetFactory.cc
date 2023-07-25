#include "MachDatasetFactory.h"
#include <data/src/TensorConversion.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <stdexcept>

namespace thirdai::automl {

data::LoaderPtr MachDatasetFactory::getDataLoader(
    const dataset::DataSourcePtr& data_source, bool include_mach_labels,
    dataset::DatasetShuffleConfig shuffle_config, bool verbose) {
  return getDataLoaderHelper(data_source, include_mach_labels, shuffle_config,
                             verbose, nullptr);
}

thirdai::data::LoaderPtr MachDatasetFactory::getColdStartDataLoader(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool include_mach_labels,
    bool fast_approximation, dataset::DatasetShuffleConfig shuffle_config,
    bool verbose) {
  thirdai::data::TransformationPtr cold_start_transformation;
  if (fast_approximation) {
    std::vector<std::string> all_columns = weak_column_names;
    all_columns.insert(all_columns.end(), strong_column_names.begin(),
                       strong_column_names.end());
    cold_start_transformation = std::make_shared<thirdai::data::StringConcat>(
        all_columns, _cold_start_text_column);
  } else {
    cold_start_transformation =
        std::make_shared<thirdai::data::ColdStartTextAugmentation>(
            /* strong_column_names= */ strong_column_names,
            /* weak_column_names= */ weak_column_names,
            /* label_column_name= */ _cold_start_label_column,
            /* output_column_name= */ _cold_start_text_column);
  }

  return getDataLoaderHelper(data_source, include_mach_labels, shuffle_config,
                             verbose, cold_start_transformation);
}

thirdai::data::LoaderPtr MachDatasetFactory::getDataLoaderHelper(
    const dataset::DataSourcePtr& data_source, bool include_mach_labels,
    dataset::DatasetShuffleConfig shuffle_config, bool verbose,
    thirdai::data::TransformationPtr cold_start_transformation) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  data::ColumnMapIterator data_iter(data_source, _delimiter);

  std::vector<data::TransformationPtr> transformations;
  if (cold_start_transformation) {
    transformations.push_back(std::move(cold_start_transformation));
  }
  transformations.push_back(_input_transformation);
  transformations.push_back(_entity_id_transformation);
  if (include_mach_labels) {
    transformations.push_back(_mach_label_transformation);
  }
  auto transformation_list = data::TransformationList::make({transformations});

  data::IndexValueColumnList label_column_outputs;
  if (include_mach_labels) {
    label_column_outputs.emplace_back(_mach_label_column, std::nullopt);
  }
  label_column_outputs.emplace_back(_entity_id_column, std::nullopt);

  return data::Loader::make(
      data_iter, transformation_list, _state, {{_input_column, std::nullopt}},
      label_column_outputs, shuffle_config.min_buffer_size, verbose);
}

bolt::nn::tensor::TensorList MachDatasetFactory::featurizeInput(
    const MapInput& sample) {
  auto columns = data::ColumnMap::fromMapInput(sample);

  columns = _input_transformation->apply(columns, *_state);

  return data::convertToTensorBatch(columns, {{_input_column, std::nullopt}});
}

bolt::nn::tensor::TensorList MachDatasetFactory::featurizeInputBatch(
    const MapInputBatch& samples) {
  auto columns = data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transformation->apply(columns, *_state);

  return data::convertToTensorBatch(columns, {{_input_column, std::nullopt}});
}

std::pair<TensorList, TensorList> MachDatasetFactory::featurizeTrainingBatch(
    const MapInputBatch& samples, bool prehashed) {
  auto columns = data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transformation->apply(columns, *_state);

  if (prehashed) {
    columns = _prehashed_label_transformation->apply(columns, *_state);
  } else {
    columns = _entity_id_transformation->apply(columns, *_state);
    columns = _mach_label_transformation->apply(columns, *_state);
  }

  auto data =
      data::convertToTensorBatch(columns, {{_input_column, std::nullopt}});

  auto labels = data::convertToTensorBatch(
      columns,
      {{_mach_label_column, std::nullopt}, {_entity_id_column, std::nullopt}});

  return std::make_pair(std::move(data), std::move(labels));
}

}  // namespace thirdai::automl