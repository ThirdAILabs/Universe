#include "Graph.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <exception>
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

ar::ConstArchivePtr GraphBuilder::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("node_id_column", ar::str(_node_id_column));
  map->set("neighbors_column", ar::str(_neighbors_column));
  map->set("feature_columns", ar::vecStr(_feature_columns));

  return map;
}

GraphBuilder::GraphBuilder(const ar::Archive& archive)
    : _node_id_column(archive.str("node_id_column")),
      _neighbors_column(archive.str("neighbors_column")),
      _feature_columns(archive.getAs<ar::VecStr>("feature_columns")) {}

template void GraphBuilder::serialize(cereal::BinaryInputArchive&);
template void GraphBuilder::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void GraphBuilder::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _node_id_column,
          _neighbors_column, _feature_columns);
}

NeighborIds::NeighborIds(std::string node_id_column,
                         std::string output_neighbors_column)
    : _node_id_column(std::move(node_id_column)),
      _output_neighbors_column(std::move(output_neighbors_column)) {}

ColumnMap NeighborIds::apply(ColumnMap columns, State& state) const {
  auto node_ids = columns.getValueColumn<uint32_t>(_node_id_column);

  const auto& graph = state.graph();

  std::vector<std::vector<uint32_t>> neighbors(columns.numRows());

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(graph, node_ids, neighbors, error) if (columns.numRows() > 1)
  for (size_t i = 0; i < node_ids->numRows(); i++) {
    try {
      for (auto nbr : graph->neighbors(node_ids->value(i))) {
        neighbors[i].push_back(nbr);
      }
      if (neighbors[i].empty()) {
        // This is a special token for no neighbors, it improves performance by
        // allowing the model to learn when there are no neighbors present. It
        // also prevents issues with bolt having undefined outputs from the
        // first layer when you pass in an empty sparse input.
        neighbors[i].push_back(std::numeric_limits<uint32_t>::max() - 1);
      }
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  auto neighbors_col = ArrayColumn<uint32_t>::make(
      std::move(neighbors), std::numeric_limits<uint32_t>::max());
  columns.setColumn(_output_neighbors_column, neighbors_col);

  return columns;
}

ar::ConstArchivePtr NeighborIds::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("node_id_column", ar::str(_node_id_column));
  map->set("output_column", ar::str(_output_neighbors_column));

  return map;
}

NeighborIds::NeighborIds(const ar::Archive& archive)
    : _node_id_column(archive.str("node_id_column")),
      _output_neighbors_column(archive.str("output_column")) {}

template void NeighborIds::serialize(cereal::BinaryInputArchive&);
template void NeighborIds::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void NeighborIds::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _node_id_column,
          _output_neighbors_column);
}

NeighborFeatures::NeighborFeatures(std::string node_id_column,
                                   std::string output_feature_column)
    : _node_id_column(std::move(node_id_column)),
      _output_features_column(std::move(output_feature_column)) {}

ColumnMap NeighborFeatures::apply(ColumnMap columns, State& state) const {
  auto node_ids = columns.getValueColumn<uint32_t>(_node_id_column);

  const auto& graph = state.graph();

  std::vector<std::vector<float>> features(columns.numRows());

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(node_ids, features, graph, error) if (columns.numRows() > 1)
  for (size_t i = 0; i < node_ids->numRows(); i++) {
    try {
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

      if (total != 0) {  // To prevent division by zero.
        for (float& feature : sum_nbr_features) {
          feature /= total;
        }
      }

      features[i] = std::move(sum_nbr_features);
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto features_col =
      ArrayColumn<float>::make(std::move(features), graph->featureDim());
  columns.setColumn(_output_features_column, features_col);

  return columns;
}

ar::ConstArchivePtr NeighborFeatures::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("node_id_column", ar::str(_node_id_column));
  map->set("output_column", ar::str(_output_features_column));

  return map;
}

NeighborFeatures::NeighborFeatures(const ar::Archive& archive)
    : _node_id_column(archive.str("node_id_column")),
      _output_features_column(archive.str("output_column")) {}

template void NeighborFeatures::serialize(cereal::BinaryInputArchive&);
template void NeighborFeatures::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void NeighborFeatures::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _node_id_column,
          _output_features_column);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::GraphBuilder)
CEREAL_REGISTER_TYPE(thirdai::data::NeighborIds)
CEREAL_REGISTER_TYPE(thirdai::data::NeighborFeatures)