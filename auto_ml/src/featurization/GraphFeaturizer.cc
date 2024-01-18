#include "GraphFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Graph.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <dataset/src/utils/GraphInfo.h>
#include <optional>
#include <stdexcept>

namespace thirdai::automl {

GraphFeaturizer::GraphFeaturizer(const ColumnDataTypes& data_types,
                                 const std::string& target_col,
                                 uint32_t n_target_classes,
                                 const TabularOptions& options)
    : _delimiter(options.delimiter) {
  auto [input_transforms, output_cols] =
      nonTemporalTransformations(data_types, target_col, options);

  auto [node_id, node_id_output] = nodeId(data_types);
  input_transforms.push_back(node_id);

  auto [nbr_ids, nbr_ids_output] = neighborIds(node_id_output);
  input_transforms.push_back(nbr_ids);

  auto [nbr_features, nbr_features_output] = neighborFeatures(node_id_output);
  input_transforms.push_back(nbr_features);
  output_cols.push_back(nbr_features_output);

  auto fh = std::make_shared<data::FeatureHash>(
      output_cols, FEATURIZED_INDICES, FEATURIZED_VALUES,
      udt::defaults::FEATURE_HASH_RANGE);
  input_transforms.push_back(fh);

  _input_transform = data::Pipeline::make(input_transforms);

  _bolt_input_columns = {
      data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES),
      data::OutputColumns(nbr_ids_output)};

  _label_transform = std::make_shared<data::StringToToken>(
      target_col, FEATURIZED_LABELS, n_target_classes);

  _bolt_label_columns = {data::OutputColumns(FEATURIZED_LABELS)};

  auto [graph_builder, graph_info] = graphBuilder(data_types);
  // clang-tidy things this can be done in the member initialization above.
  _graph_builder = graph_builder;  // NOLINT

  _state = std::make_shared<data::State>(graph_info);
}

data::LoaderPtr GraphFeaturizer::indexAndGetDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  index(data_source);

  data_source->restart();

  auto data_iter = data::CsvIterator::make(data_source, _delimiter);

  auto transformation_list =
      data::Pipeline::make({_input_transform, _label_transform});

  return data::Loader::make(
      data_iter, transformation_list, _state, _bolt_input_columns,
      _bolt_label_columns, /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_buffer_size= */ shuffle_config.min_buffer_size,
      /* shuffle_seed= */ shuffle_config.seed);
}

void GraphFeaturizer::index(const dataset::DataSourcePtr& data_source) {
  auto data_iter = data::CsvIterator::make(data_source, _delimiter);

  while (auto chunk = data_iter->next()) {
    _graph_builder->apply(*chunk, *_state);
  }
}

bolt::TensorList GraphFeaturizer::featurizeInput(const MapInput& sample) {
  auto columns = data::ColumnMap::fromMapInput(sample);

  columns = _input_transform->apply(std::move(columns), *_state);

  return data::toTensors(columns, _bolt_input_columns);
}

bolt::TensorList GraphFeaturizer::featurizeInputBatch(
    const MapInputBatch& samples) {
  auto columns = data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform->apply(std::move(columns), *_state);

  return data::toTensors(columns, _bolt_input_columns);
}

std::string neighborsColumn(const ColumnDataTypes& data_types) {
  std::optional<std::string> neighbors_col = std::nullopt;
  for (const auto& [col_name, data_type] : data_types) {
    if (asNeighbors(data_type)) {
      if (neighbors_col) {
        throw std::invalid_argument(
            "Only a single neighbors column is allowed in GNN.");
      }
      neighbors_col = col_name;
    }
  }

  if (!neighbors_col) {
    throw std::invalid_argument("Neighbors column is required for GNN.");
  }
  return *neighbors_col;
}

std::string nodeIdColumn(const ColumnDataTypes& data_types) {
  std::optional<std::string> node_id_column = std::nullopt;
  for (const auto& [col_name, data_type] : data_types) {
    if (asNodeID(data_type)) {
      if (node_id_column) {
        throw std::invalid_argument(
            "Only a single node ID column is allowed in GNN.");
      }
      node_id_column = col_name;
    }
  }

  if (!node_id_column) {
    throw std::invalid_argument("NodeID column is required for GNN.");
  }
  return *node_id_column;
}

std::pair<data::TransformationPtr, std::string> GraphFeaturizer::nodeId(
    const ColumnDataTypes& data_types) {
  for (const auto& [col_name, data_type] : data_types) {
    if (asNodeID(data_type)) {
      return {std::make_shared<data::StringToToken>(col_name, col_name,
                                                    std::nullopt),
              col_name};
    }
  }
  throw std::invalid_argument("NodeID column is required for GNN.");
}

std::pair<data::TransformationPtr, std::string>
GraphFeaturizer::neighborFeatures(const std::string& nod_id_col) {
  auto transform =
      std::make_shared<data::NeighborFeatures>(nod_id_col, GRAPH_NBR_FEATURES);

  return {transform, GRAPH_NBR_FEATURES};
}

std::pair<data::TransformationPtr, std::string> GraphFeaturizer::neighborIds(
    const std::string& nod_id_col) {
  auto transform =
      std::make_shared<data::NeighborIds>(nod_id_col, GRAPH_NBR_IDS);

  return {transform, GRAPH_NBR_IDS};
}

std::pair<data::TransformationPtr, GraphInfoPtr> GraphFeaturizer::graphBuilder(
    const ColumnDataTypes& data_types) {
  std::vector<std::string> feature_col_names;

  std::vector<data::TransformationPtr> transforms;

  for (const auto& [col_name, data_type] : data_types) {
    if (asNumerical(data_type)) {
      feature_col_names.push_back(col_name);
      transforms.push_back(
          std::make_shared<data::StringToDecimal>(col_name, col_name));
    }
  }

  auto [node_id_transform, node_id_col] = nodeId(data_types);

  transforms.push_back(node_id_transform);

  std::string nbrs_column = neighborsColumn(data_types);

  auto parse_nbrs = std::make_shared<data::StringToTokenArray>(
      nbrs_column, nbrs_column, ' ', std::nullopt);
  transforms.push_back(parse_nbrs);

  auto graph_builder = std::make_shared<data::GraphBuilder>(
      node_id_col, nbrs_column, feature_col_names);
  transforms.push_back(graph_builder);

  auto graph_info = std::make_shared<GraphInfo>(feature_col_names.size());

  return {data::Pipeline::make(transforms), graph_info};
}

template void GraphFeaturizer::serialize(cereal::BinaryInputArchive&);
template void GraphFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void GraphFeaturizer::serialize(Archive& archive) {
  archive(_input_transform, _label_transform, _graph_builder,
          _bolt_input_columns, _bolt_label_columns, _delimiter, _state);
}

}  // namespace thirdai::automl