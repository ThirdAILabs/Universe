#include "ColdStartUtils.h"
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <stdexcept>

namespace thirdai::automl::cold_start {

dataset::cold_start::ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::TabularDatasetFactoryPtr& dataset_factory) {

  if (!dataset_factory->integerTarget()) {
    throw std::invalid_argument(
        "Cold start pretraining currently only supports integer labels.");
  }

  if (dataset_factory->inputDataTypes().size() != 1 ||
      !data::asText(dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "Cold start pretraining can only be used on datasets with a single "
        "text input column and target column. The current model is configured "
        "with " +
        std::to_string(dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }

  std::string text_column_name =
      dataset_factory->inputDataTypes().begin()->first;

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      data, dataset_factory->delimiter());

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ dataset_factory->labelColumn(),
      /* output_column_name= */ text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto data_source = thirdai::dataset::cold_start::ColdStartDataSource::make(
      /* column_map= */ augmented_data,
      /* text_column_name= */ text_column_name,
      /* label_column_name= */ dataset_factory->labelColumn(),
      /* column_delimiter= */ dataset_factory->delimiter(),
      /* label_delimiter= */ dataset_factory->labelDelimiter(),
      /* resource_name = */ data->resourceName());

  return data_source;
}
}  // namespace thirdai::automl::cold_start