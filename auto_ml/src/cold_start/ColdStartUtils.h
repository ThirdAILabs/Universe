#pragma once

#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>

namespace thirdai::automl::cold_start {

struct ColdStartMetadata {
  std::string text_column_name;
  std::optional<char> label_delimiter;
};

// Verifies that the model is cold start compatible and returns metadata needed
// for the cold start pretraining.
ColdStartMetadata getColdStartMetadata(
    const data::UDTConfigPtr& dataset_config);

ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& original_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::UDTConfigPtr& dataset_config);

}  // namespace thirdai::automl::cold_start