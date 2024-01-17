#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::automl {

// TODO(Nicholas): Remove the data subnamespace from automl so it doesn't clash
// with data.

std::pair<data::TransformationPtr, data::OutputColumnsList>
inputTransformations(const ColumnDataTypes& data_types,
                     const std::string& label_column,
                     const TemporalRelationships& temporal_relationships,
                     const TabularOptions& options, bool should_update_history);

// This represents a sequence of transformations and the final output columns.
using MergedTransformSeries =
    std::pair<std::vector<data::TransformationPtr>, std::vector<std::string>>;

MergedTransformSeries nonTemporalTransformations(
    ColumnDataTypes data_types, const std::string& label_column,
    const TabularOptions& options);

}  // namespace thirdai::automl