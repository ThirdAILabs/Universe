#pragma once

#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>

namespace thirdai::automl::cold_start {

// Verifies that the model has the correct number of columns it expects in the
// dataset and that the target column is an integer.
void verifyTaskIsColdStartCompatible(const data::UDTConfigPtr& dataset_config);

// Verifies that the model has a text input and returns the name of the text
// column.
std::string verifyTextColumn(const data::UDTConfigPtr& dataset_config);

// Verifies that the target is categorical and returns the delimiter.
std::optional<char> verifyCategoricalTarget(
    const data::UDTConfigPtr& dataset_config);

// Checks that the label column is a TokenArrayColumn. If it is not then it will
// attempt to convert it to a TokenArrayColumn if possible.
void verifyLabelColumnIsTokenArray(thirdai::data::ColumnMap& columns,
                                   const std::string& label_column_name);

}  // namespace thirdai::automl::cold_start