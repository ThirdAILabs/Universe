#include "ColdStartUtils.h"
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <stdexcept>

namespace thirdai::automl::cold_start {

// Verifies that the model has the correct number of columns it expects in the
// dataset and that the target column is an integer.
void verifyTaskIsColdStartCompatible(const data::UDTConfigPtr& dataset_config) {
  if (!dataset_config->integer_target) {
    throw std::invalid_argument(
        "Cold start pretraining currently only supports integer labels.");
  }

  if (dataset_config->data_types.size() != 2) {
    throw std::invalid_argument(
        "Cold start pretraining can only be used on datasets with a single "
        "text input column and target column. The current model is configured "
        "with " +
        std::to_string(dataset_config->data_types.size()) + " columns.");
  }
}

/**
 * Verifies that the model has a text input and returns the name of the text
 * column. This is epected to run along with verifyTaskIsColdStartCompatible and
 * so it returns the first text column it encounters because
 * verifyTaskIsColdStartCompatible will ensure that there are only 2 columns, so
 * the non target column must be the single text column.
 */
std::string getTextColumnName(const data::UDTConfigPtr& dataset_config) {
  std::string text_column_name;
  for (const auto& [name, meta] : dataset_config->data_types) {
    if (name != dataset_config->target && data::asText(meta)) {
      return name;
    }
  }

  throw std::invalid_argument(
      "Non target column must be text to use cold start.");
}

// Verifies that the target is categorical and returns the label delimiter.
std::optional<char> getLabelDelimiter(
    const data::UDTConfigPtr& dataset_config) {
  if (auto metadata = data::asCategorical(
          dataset_config->data_types.at(dataset_config->target))) {
    return metadata->delimiter;
  }
  throw std::invalid_argument(
      "Cold start pretraining is only supported for classification tasks.");
}

ColdStartMetadata getColdStartMetadata(
    const data::UDTConfigPtr& dataset_config) {
  verifyTaskIsColdStartCompatible(dataset_config);

  ColdStartMetadata metadata{getTextColumnName(dataset_config),
                             getLabelDelimiter(dataset_config)};

  return metadata;
}
dataset::ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& original_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::UDTConfigPtr& dataset_config) {
  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      original_source, dataset_config->delimiter);

  auto metadata = cold_start::getColdStartMetadata(dataset_config);

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ dataset_config->target,
      /* output_column_name= */ metadata.text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto data_source = dataset::ColdStartDataSource::make(
      /* column_map= */ augmented_data,
      /* text_column_name= */ metadata.text_column_name,
      /* label_column_name= */ dataset_config->target,
      /* column_delimiter= */ dataset_config->delimiter,
      /* label_delimiter= */ metadata.label_delimiter,
      /* resource_name = */ original_source->resourceName());

  return data_source;
}

}  // namespace thirdai::automl::cold_start