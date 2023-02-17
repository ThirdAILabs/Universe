#include "GraphDatasetFactory.h"
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/DatasetFactoryUtils.h>
#include <auto_ml/src/models/UDTUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/GraphBlocks.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <stdexcept>

namespace thirdai::automl::data {

std::pair<GraphInfoPtr, dataset::BlockPtr> createGraphInfoAndBuilder(
    const data::ColumnDataTypes& data_types) {
  std::vector<std::string> feature_col_names;
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

/*
 * Pops and returns the NeighborTokensBlock from the passed in block list.
 * The block list must have a single NeighborTokensBlock.
 */

dataset::BlockPtr popNeighborTokensBlock(
    std::vector<dataset::BlockPtr> blocks) {
  int64_t neighbor_tokens_block_index = -1;
  for (size_t block_id = 0; block_id < blocks.size(); block_id++) {
    if (dynamic_cast<dataset::NeighborTokensBlock*>(
            blocks.at(block_id).get())) {
      neighbor_tokens_block_index = block_id;
      break;
    }
  }

  if (neighbor_tokens_block_index < 0) {
    throw std::logic_error(
        "The passed in block list should have a NeighborTokensBlock");
  }

  dataset::BlockPtr neighbor_tokens_block =
      blocks.at(neighbor_tokens_block_index);
  blocks.erase(blocks.begin() + neighbor_tokens_block_index);

  return neighbor_tokens_block;
}

GraphDatasetFactory::GraphDatasetFactory(data::ColumnDataTypes data_types,
                                         std::string target_col,
                                         uint32_t n_target_classes,
                                         char delimiter, uint32_t max_neighbors,
                                         uint32_t k_hop,
                                         bool store_node_features)
    : _data_types(std::move(data_types)),
      _target_col(std::move(target_col)),
      _n_target_classes(n_target_classes),
      _delimiter(delimiter),
      _max_neighbors(max_neighbors),
      _k_hop(k_hop),
      _store_node_features(store_node_features) {
  verifyExpectedNumberOfGraphTypes(data_types, /* expected_count = */ 1);

  if (_k_hop > 3 || _k_hop == 0) {
    throw std::invalid_argument("K hop must be between 1 and 3 inclusive.");
  }

  auto [graph_info, graph_builder_block] =
      createGraphInfoAndBuilder(data_types);

  std::vector<dataset::BlockPtr> feature_blocks =
      FeatureComposer::makeNonTemporalFeatureBlocks(
          data_types, target_col,
          /* temporal_relationships = */ TemporalRelationships(),
          /* vectors_map = */ PreprocessedVectorsMap(),
          /* text_pairgrams_word_limit = */ models::TEXT_PAIRGRAM_WORD_LIMIT,
          /* contextual_columns = */ true, /* graph_info = */ graph_info);

  dataset::BlockPtr label_block = dataset::NumericalCategoricalBlock::make(
      /* col = */ target_col,
      /* n_classes= */ n_target_classes);

  dataset::BlockPtr sparse_neighbor_block =
      popNeighborTokensBlock(feature_blocks);

  _featurizer = dataset::TabularFeaturizer::make(
      /* block_lists = */ {dataset::BlockList(
                               std::move(feature_blocks),
                               /* hash_range = */ DEFAULT_HASH_RANGE),
                           dataset::BlockList({sparse_neighbor_block}),
                           dataset::BlockList({label_block})},
      /* has_header= */ true,
      /* delimiter= */ delimiter, /* parallel= */ true,
      /* hash_range= */ DEFAULT_HASH_RANGE);

  _graph_builder = dataset::TabularFeaturizer::make(
      /* blocks = */ {dataset::BlockList({graph_builder_block})},
      /* has_header= */ true,
      /* delimiter= */ delimiter, /* parallel= */ true,
      /* hash_range= */ DEFAULT_HASH_RANGE);

  _graph_info = graph_info;
}

dataset::DatasetLoaderPtr GraphDatasetFactory::getLabeledDatasetLoader(
    std::shared_ptr<dataset::DataSource> data_source, bool training) {
  auto column_number_map =
      makeColumnNumberMapFromHeader(*data_source, _delimiter);

  // TODO(Josh): Abstract this
  std::vector<std::string> column_number_to_name =
      column_number_map.getColumnNumToColNameMap();

  // The featurizer will treat the next line as a header
  // Restart so featurizer does not skip a sample.
  data_source->restart();

  _featurizer->updateColumnNumbers(column_number_map);
  _graph_builder->updateColumnNumbers(column_number_map);

  // If we want to save memory by not storing node features, we clear it here
  if (!_store_node_features) {
    _graph_info->clear();
  }
  dataset::DatasetLoader graph_builder_loader(data_source, _graph_builder,
                                              /* shuffle = */ false);
  graph_builder_loader.loadAll(
      /* batch_size = */ DEFAULT_INTERNAL_FEATURIZATION_BATCH_SIZE);

  // The featurizer will treat the next line as a header
  // Restart so featurizer does not skip a sample.
  data_source->restart();

  return std::make_unique<dataset::DatasetLoader>(data_source, _featurizer,
                                                  /* shuffle= */ training);
}

}  // namespace thirdai::automl::data