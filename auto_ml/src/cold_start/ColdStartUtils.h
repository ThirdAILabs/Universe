#pragma once

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

// Checks that the label column is a TokenArrayColumn. If it is not then it will
// attempt to convert it to a TokenArrayColumn if possible.
void convertLabelColumnToTokenArray(thirdai::data::ColumnMap& columns,
                                    const std::string& label_column_name,
                                    std::optional<char> label_delimiter);

}  // namespace thirdai::automl::cold_start