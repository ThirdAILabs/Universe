#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::automl {

// TODO(Nicholas): Remove the data subnamespace from automl so it doesn't clash
// with thirdai::data.

std::pair<thirdai::data::TransformationPtr, thirdai::data::IndexValueColumnList>
inputTransformations(const data::ColumnDataTypes& data_types,
                     const std::string& label_column,
                     const data::TemporalRelationships& temporal_relationships,
                     const data::TabularOptions& options,
                     bool should_update_history);

using CreatedTransformations =
    std::pair<std::vector<thirdai::data::TransformationPtr>,
              std::vector<std::string>>;

CreatedTransformations nonTemporalTransformations(
    data::ColumnDataTypes data_types, const std::string& label_column,
    const data::TabularOptions& options);

}  // namespace thirdai::automl