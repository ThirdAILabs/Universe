#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/featurization/TemporalContext.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/PreprocessedVectors.h>

namespace thirdai::automl {

using PreprocessedVectorsMap =
    std::unordered_map<std::string, dataset::PreprocessedVectorsPtr>;

std::vector<dataset::BlockPtr> makeTabularInputBlocks(
    const ColumnDataTypes& input_data_types,
    const std::set<std::string>& label_col_names,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    TemporalContext& temporal_context, bool should_update_history,
    const TabularOptions& options);

std::vector<dataset::BlockPtr> makeNonTemporalInputBlocks(
    const ColumnDataTypes& input_data_types,
    const std::set<std::string>& label_col_names,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map, const TabularOptions& options);

std::vector<dataset::BlockPtr> makeTemporalInputBlocks(
    const ColumnDataTypes& input_data_types,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    TemporalContext& temporal_context, bool should_update_history,
    const TabularOptions& options);

}  // namespace thirdai::automl