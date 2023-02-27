#include "GraphDatasetManager.h"
#include "FeaturizationUtils.h"
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/GraphBlocks.h>
#include <dataset/src/blocks/TabularHashFeatures.h>

namespace thirdai::automl::data {

std::pair<GraphInfoPtr, dataset::BlockPtr> createGraphInfoAndBuilder(
    const data::ColumnDataTypes& data_types);

/*
 * Pops and returns the NeighborTokensBlock from the passed in block list.
 * The block list must have a single NeighborTokensBlock.
 */
dataset::BlockPtr popNeighborTokensBlock(
    std::vector<dataset::BlockPtr>& blocks);

GraphDatasetManager::GraphDatasetManager(data::ColumnDataTypes data_types,
                                         std::string target_col,
                                         uint32_t n_target_classes,
                                         char delimiter, bool use_pairgrams)
    : _data_types(std::move(data_types)),
      _target_col(std::move(target_col)),
      _n_target_classes(n_target_classes),
      _delimiter(delimiter) {
  utils::verifyExpectedNumberOfGraphTypes(_data_types,
                                          /* expected_count = */ 1);

  dataset::BlockPtr graph_builder_block;
  std::tie(_graph_info, graph_builder_block) =
      createGraphInfoAndBuilder(_data_types);

  std::vector<dataset::BlockPtr> feature_blocks =
      FeatureComposer::makeNonTemporalFeatureBlocks(
          _data_types, _target_col,
          /* temporal_relationships = */ TemporalRelationships(),
          /* vectors_map = */ PreprocessedVectorsMap(),
          /* text_pairgrams_word_limit = */ udt::defaults::PAIRGRAM_WORD_LIMIT,
          /* contextual_columns = */ use_pairgrams,
          /* graph_info = */ _graph_info);

  dataset::BlockPtr sparse_neighbor_block =
      popNeighborTokensBlock(feature_blocks);

  dataset::BlockPtr label_block = dataset::NumericalCategoricalBlock::make(
      /* col = */ _target_col,
      /* n_classes= */ n_target_classes);

  _featurizer = dataset::TabularFeaturizer::make(
      /* block_lists = */ {dataset::BlockList(std::move(feature_blocks),
                                              /* hash_range = */ udt::defaults::
                                                  FEATURE_HASH_RANGE),
                           dataset::BlockList({sparse_neighbor_block}),
                           dataset::BlockList({label_block})},
      /* has_header= */ true,
      /* delimiter= */ delimiter, /* parallel= */ true);

  _graph_builder = dataset::TabularFeaturizer::make(
      /* blocks = */ {dataset::BlockList({graph_builder_block})},
      /* has_header= */ true,
      /* delimiter= */ delimiter, /* parallel= */ true);
}

dataset::DatasetLoaderPtr GraphDatasetManager::indexAndGetDatasetLoader(
    const dataset::DataSourcePtr& data_source) {
  index(data_source);

  data_source->restart();

  utils::updateFeaturizerWithHeader(_featurizer, data_source, _delimiter);

  return std::make_unique<dataset::DatasetLoader>(data_source, _featurizer,
                                                  /* shuffle= */ true);
}

void GraphDatasetManager::index(const dataset::DataSourcePtr& data_source) {
  utils::updateFeaturizerWithHeader(_graph_builder, data_source, _delimiter);

  dataset::DatasetLoader graph_builder_loader(data_source, _graph_builder,
                                              /* shuffle = */ false);

  graph_builder_loader.loadAll(
      /* batch_size = */ dataset::DEFAULT_FEATURIZATION_BATCH_SIZE);
}

std::pair<GraphInfoPtr, dataset::BlockPtr> createGraphInfoAndBuilder(
    const data::ColumnDataTypes& data_types) {
  std::vector<dataset::ColumnIdentifier> feature_col_names;
  std::string neighbor_col_name, node_id_col_name;

  for (const auto& [col_name, data_type] : data_types) {
    if (asNeighbors(data_type)) {
      neighbor_col_name = col_name;
    } else if (asNodeID(data_type)) {
      node_id_col_name = col_name;
    } else if (asNumerical(data_type)) {
      feature_col_names.push_back(col_name);
    }
  }

  GraphInfoPtr graph_info =
      std::make_shared<GraphInfo>(/* feature_dim = */ feature_col_names.size());

  dataset::BlockPtr builder_block = dataset::GraphBuilderBlock::make(
      neighbor_col_name, node_id_col_name, feature_col_names, graph_info);

  return {graph_info, builder_block};
}

dataset::BlockPtr popNeighborTokensBlock(
    std::vector<dataset::BlockPtr>& blocks) {
  std::optional<uint64_t> neighbor_tokens_block_index = -1;
  for (size_t block_id = 0; block_id < blocks.size(); block_id++) {
    if (dynamic_cast<dataset::NeighborTokensBlock*>(
            blocks.at(block_id).get())) {
      neighbor_tokens_block_index = block_id;
      break;
    }
  }

  if (!neighbor_tokens_block_index.has_value()) {
    throw std::logic_error(
        "The passed in block list should have a NeighborTokensBlock");
  }

  dataset::BlockPtr neighbor_tokens_block =
      blocks.at(*neighbor_tokens_block_index);
  blocks.erase(blocks.begin() + *neighbor_tokens_block_index);

  return neighbor_tokens_block;
}

}  // namespace thirdai::automl::data