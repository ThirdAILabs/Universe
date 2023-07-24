#include "MachDatasetFactory.h"
#include <data/src/TensorConversion.h>
#include <data/src/transformations/TransformationList.h>
#include <stdexcept>

namespace thirdai::automl {

data::LoaderPtr MachDatasetFactory::getDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size,
    size_t max_batches, bool include_mach_labels,
    dataset::DatasetShuffleConfig shuffle_config, bool verbose) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  data::ColumnMapIterator data_iter(data_source, _delimiter);

  std::vector<data::TransformationPtr> transformations;
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

  return data::Loader::make(data_iter, transformation_list, _state,
                            {{_input_column, std::nullopt}},
                            label_column_outputs, batch_size, max_batches,
                            shuffle_config.min_buffer_size, verbose);
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