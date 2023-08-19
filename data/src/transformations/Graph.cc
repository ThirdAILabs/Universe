#include "Graph.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace thirdai::data {

GraphBuilder::GraphBuilder(std::string node_id_column,
                           std::string neighbors_column,
                           std::vector<std::string> feature_columns)
    : _node_id_column(std::move(node_id_column)),
      _neighbors_column(std::move(neighbors_column)),
      _feature_columns(std::move(feature_columns)) {}

ColumnMap GraphBuilder::apply(ColumnMap columns, State& state) const {
  std::vector<ValueColumnBasePtr<float>> features;
  for (const auto& col : _feature_columns) {
    features.push_back(columns.getValueColumn<float>(col));
  }

  std::vector<std::vector<float>> dense_feature_vectors(columns.numRows());
  for (const auto& feature : features) {
    for (size_t i = 0; i < columns.numRows(); i++) {
      dense_feature_vectors[i].push_back(feature->value(i));
    }
  }

  auto node_ids = columns.getValueColumn<uint32_t>(_node_id_column);
  auto neighbors = columns.getArrayColumn<uint32_t>(_neighbors_column);

  const auto& graph = state.graph();

  if (graph->featureDim() != features.size()) {
    throw std::runtime_error(
        "Graph feature dim does not match number of feature columns.");
  }

  for (size_t i = 0; i < columns.numRows(); i++) {
    uint32_t node_id = node_ids->value(i);

    std::vector<uint64_t> neighbor_list;
    for (auto nbr : neighbors->row(i)) {
      neighbor_list.push_back(nbr);
    }

    graph->insertNode(node_id, std::move(dense_feature_vectors[i]),
                      std::move(neighbor_list));
  }

  return columns;
}

NeighborIds::NeighborIds(std::string node_id_column,
                         std::string output_neighbors_column)
    : _node_id_column(std::move(node_id_column)),
      _output_neighbors_column(std::move(output_neighbors_column)) {}

ColumnMap NeighborIds::apply(ColumnMap columns, State& state) const {
  auto node_ids = columns.getValueColumn<uint32_t>(_node_id_column);

  const auto& graph = state.graph();

  std::vector<std::vector<uint32_t>> neighbors(columns.numRows());

#pragma omp parallel for default(none) shared(graph, node_ids, neighbors)
  for (size_t i = 0; i < node_ids->numRows(); i++) {
    for (auto nbr : graph->neighbors(node_ids->value(i))) {
      neighbors[i].push_back(nbr);
    }
  }

  auto neighbors_col = ArrayColumn<uint32_t>::make(
      std::move(neighbors), std::numeric_limits<uint32_t>::max());
  columns.setColumn(_output_neighbors_column, neighbors_col);

  return columns;
}

NeighborFeatures::NeighborFeatures(std::string node_id_column,
                                   std::string output_feature_column)
    : _node_id_column(std::move(node_id_column)),
      _output_features_column(std::move(output_feature_column)) {}

ColumnMap NeighborFeatures::apply(ColumnMap columns, State& state) const {
  auto node_ids = columns.getValueColumn<uint32_t>(_node_id_column);

  const auto& graph = state.graph();

  std::vector<std::vector<float>> features(columns.numRows());

#pragma omp parallel for default(none) shared(node_ids, features, graph)
  for (size_t i = 0; i < node_ids->numRows(); i++) {
    uint32_t node_id = node_ids->value(i);

    std::vector<float> sum_nbr_features(graph->featureDim());

    for (auto nbr : graph->neighbors(node_id)) {
      const auto& nbr_features = graph->featureVector(nbr);
      for (size_t j = 0; j < nbr_features.size(); j++) {
        sum_nbr_features.at(j) += nbr_features.at(j);
      }
    }

    // Normalize neighbor features.
    float total =
        std::reduce(sum_nbr_features.begin(), sum_nbr_features.end(), 0.0);
    for (float& feature : sum_nbr_features) {
      feature /= total;
    }

    features[i] = std::move(sum_nbr_features);
  }

  auto features_col =
      ArrayColumn<float>::make(std::move(features), graph->featureDim());
  columns.setColumn(_output_features_column, features_col);

  return columns;
}

}  // namespace thirdai::data