#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/TemporalContext.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/PreprocessedVectors.h>

namespace thirdai::automl::data::tabular {

using PreprocessedVectorsMap =
    std::unordered_map<std::string, dataset::PreprocessedVectorsPtr>;

struct TabularBlockOptions {
  uint32_t text_pairgrams_word_limit = 15;
  bool contextual_columns = false;
  std::string time_granularity = "d";
  uint32_t lookahead = 0;
  uint32_t feature_hash_range = 100000;
};

std::vector<dataset::BlockPtr> makeTabularInputBlocks(
    const ColumnDataTypes& input_data_types,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    TemporalContext& temporal_context, bool should_update_history,
    const TabularBlockOptions& options);

std::vector<dataset::BlockPtr> makeNonTemporalInputBlocks(
    const ColumnDataTypes& input_data_types,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    const TabularBlockOptions& options);

std::vector<dataset::BlockPtr> makeTemporalInputBlocks(
    const ColumnDataTypes& input_data_types,
    const TemporalRelationships& temporal_relationships,
    const PreprocessedVectorsMap& vectors_map,
    TemporalContext& temporal_context, bool should_update_history,
    const TabularBlockOptions& options);

}  // namespace thirdai::automl::data::tabular