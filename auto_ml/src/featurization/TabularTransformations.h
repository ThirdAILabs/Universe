#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::automl {

// TODO(Nicholas): Remove the data subnamespace from automl so it doesn't clash
// with thirdai::data.

std::pair<thirdai::data::TransformationPtr, thirdai::data::OutputColumnsList>
inputTransformations(const data::ColumnDataTypes& data_types,
                     const std::string& label_column,
                     const data::TemporalRelationships& temporal_relationships,
                     const data::TabularOptions& options,
                     bool should_update_history);

// This represents the transformations and outputs for a set of columns in the
// input.
using MergedTransformSeries =
    std::pair<std::vector<thirdai::data::TransformationPtr>,
              std::vector<std::string>>;

MergedTransformSeries nonTemporalTransformations(
    data::ColumnDataTypes data_types, const std::string& label_column,
    const data::TabularOptions& options);

}  // namespace thirdai::automl