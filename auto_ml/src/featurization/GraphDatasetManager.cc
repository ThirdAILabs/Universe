#include "GraphDatasetManager.h"
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/featurization/TabularBlockComposer.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/GraphBlocks.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <stdexcept>

namespace thirdai::automl::data {

struct GraphBlocks {
  std::shared_ptr<dataset::GraphBuilderBlock> builder_block;
  std::shared_ptr<dataset::NeighborTokensBlock> neighbor_tokens_block;
  std::shared_ptr<dataset::NormalizedNeighborVectorsBlock>
      normalized_neighbors_block;
};

std::pair<GraphInfoPtr, GraphBlocks> createGraphInfoAndGraphBlocks(
    const data::ColumnDataTypes& data_types);

GraphDatasetManager::GraphDatasetManager(data::ColumnDataTypes data_types,
                                         std::string target_col,
                                         uint32_t n_target_classes,
                                         const TabularOptions& options)
    : _data_types(std::move(data_types)),
      _target_col(std::move(target_col)),
      _n_target_classes(n_target_classes),
      _delimiter(options.delimiter) {
  GraphBlocks graph_blocks;
  std::tie(_graph_info, graph_blocks) =
      createGraphInfoAndGraphBlocks(_data_types);

  std::vector<dataset::BlockPtr> feature_blocks = makeNonTemporalInputBlocks(
      /* data_types = */ _data_types,
      /* label_col_names = */ {_target_col},
      /* temporal_relationships = */ {},
      /* vectors_map = */ {},
      /* options = */ options);
  feature_blocks.push_back(graph_blocks.normalized_neighbors_block);

  auto feature_blocklist =
      dataset::BlockList(std::move(feature_blocks),
                         /* hash_range = */ udt::defaults::FEATURE_HASH_RANGE);
  auto neighbor_tokens_blocklist =
      dataset::BlockList({graph_blocks.neighbor_tokens_block});
  auto label_blocklist =
      dataset::BlockList({dataset::NumericalCategoricalBlock::make(
          /* col = */ _target_col,
          /* n_classes= */ _n_target_classes)});

  _labeled_featurizer = dataset::TabularFeaturizer::make(
      /* block_lists = */ {feature_blocklist, neighbor_tokens_blocklist,
                           label_blocklist},
      /* has_header= */ true,
      /* delimiter= */ _delimiter, /* parallel= */ true);

  _inference_featurizer = dataset::TabularFeaturizer::make(
      /* block_lists = */ {feature_blocklist, neighbor_tokens_blocklist},
      /* has_header= */ true,
      /* delimiter= */ _delimiter, /* parallel= */ true);

  _graph_builder = dataset::TabularFeaturizer::make(
      /* blocks = */ {dataset::BlockList({graph_blocks.builder_block})},
      /* has_header= */ true,
      /* delimiter= */ _delimiter, /* parallel= */ true);
}

dataset::DatasetLoaderPtr GraphDatasetManager::indexAndGetLabeledDatasetLoader(
    const dataset::DataSourcePtr& data_source, bool shuffle) {
  index(data_source);

  data_source->restart();

  return std::make_unique<dataset::DatasetLoader>(data_source,
                                                  _labeled_featurizer, shuffle);
}

void GraphDatasetManager::index(const dataset::DataSourcePtr& data_source) {
  dataset::DatasetLoader graph_builder_loader(data_source, _graph_builder,
                                              /* shuffle = */ false);

  graph_builder_loader.loadAll(
      /* batch_size = */ dataset::DEFAULT_FEATURIZATION_BATCH_SIZE);
}

std::pair<GraphInfoPtr, GraphBlocks> createGraphInfoAndGraphBlocks(
    const data::ColumnDataTypes& data_types) {
  std::vector<dataset::ColumnIdentifier> feature_col_names;
  std::string neighbor_col_name, node_id_col_name;
  GraphBlocks graph_blocks;
  GraphInfoPtr graph_info =
      std::make_shared<GraphInfo>(/* feature_dim = */ feature_col_names.size());

  // TODO(Josh): Look in to combining non-numeric data from neighbors as well,
  // e.g. for a string column concatenating each neighbor's text.
  for (const auto& [col_name, data_type] : data_types) {
    if (asNeighbors(data_type)) {
      neighbor_col_name = col_name;
    } else if (asNodeID(data_type)) {
      node_id_col_name = col_name;
    } else if (asNumerical(data_type)) {
      feature_col_names.push_back(col_name);
    }
  }

  // TODO(Josh): Do a thorough ablation study (this block seems only marginally
  // useful on yelp). This should include looking at binning vs. non binning
  // for both these average features and the normal features.
  graph_blocks.normalized_neighbors_block =
      dataset::NormalizedNeighborVectorsBlock::make(node_id_col_name,
                                                    graph_info);
  // We could alternatively build this block with the neighbors
  // column, but using the node id column and graph_info instead allows us to
  // potentially not have to have a neighbors column for inference.
  graph_blocks.neighbor_tokens_block =
      dataset::NeighborTokensBlock::make(node_id_col_name, graph_info);

  graph_blocks.builder_block = dataset::GraphBuilderBlock::make(
      neighbor_col_name, node_id_col_name, feature_col_names, graph_info);

  return {graph_info, graph_blocks};
}

template void GraphDatasetManager::serialize(cereal::BinaryInputArchive&);
template void GraphDatasetManager::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void GraphDatasetManager::serialize(Archive& archive) {
  archive(_data_types, _target_col, _n_target_classes, _delimiter,
          _graph_builder, _labeled_featurizer, _inference_featurizer,
          _graph_info);
}

std::vector<BoltBatch> GraphDatasetManager::featurizeInputBatch(
    const dataset::MapInputBatch& inputs) {
  dataset::MapBatchRef inputs_ref(inputs);
  std::vector<std::vector<BoltVector>> batches =
      _inference_featurizer->featurize(inputs_ref);

  std::vector<BoltBatch> result;
  result.reserve(batches.size());
  for (auto& batch : batches) {
    result.emplace_back(std::move(batch));
  }

  return result;
}

std::vector<BoltVector> GraphDatasetManager::featurizeInput(
    const dataset::MapInput& input) {
  dataset::MapSampleRef input_ref(input);
  return _inference_featurizer->featurize(input_ref);
}

}  // namespace thirdai::automl::data