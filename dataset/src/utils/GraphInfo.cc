#include "GraphInfo.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <utility>

namespace thirdai::automl::data {

GraphInfo::GraphInfo(const proto::data::GraphInfo& graph)
    : _feature_dim(graph.feature_dim()) {
  for (const auto& [node_id, features] : graph.node_features()) {
    _node_id_to_feature_vector[node_id] = {features.features().begin(),
                                           features.features().end()};
  }

  for (const auto& [node_id, neighbors] : graph.neighbors()) {
    _node_id_to_neighbors[node_id] = {neighbors.neighbors().begin(),
                                      neighbors.neighbors().end()};
  }
}

const std::vector<float>& GraphInfo::featureVector(uint64_t node_id) const {
  if (!_node_id_to_feature_vector.count(node_id)) {
    throw GraphConstructionError(
        "No feature vector currently stored for node " +
        std::to_string(node_id));
  }

  return _node_id_to_feature_vector.at(node_id);
}

const std::vector<uint64_t>& GraphInfo::neighbors(uint64_t node_id) const {
  if (!_node_id_to_neighbors.count(node_id)) {
    throw GraphConstructionError(
        "No neighbors vector currently stored for node " +
        std::to_string(node_id));
  }

  return _node_id_to_neighbors.at(node_id);
}

void GraphInfo::insertNode(uint64_t node_id, std::vector<float> features,
                           std::vector<uint64_t> neighbors) {
#pragma omp critical
  {
    _node_id_to_feature_vector[node_id] = std::move(features);
    _node_id_to_neighbors[node_id] = std::move(neighbors);
  }
}

void GraphInfo::clear() {
  _node_id_to_feature_vector.clear();
  _node_id_to_neighbors.clear();
}

proto::data::GraphInfo* GraphInfo::toProto() const {
  auto* graph = new proto::data::GraphInfo();

  graph->set_feature_dim(_feature_dim);

  for (const auto& [node_id, features] : _node_id_to_feature_vector) {
    proto::data::GraphFeatureVector features_proto;
    *features_proto.mutable_features() = {features.begin(), features.end()};

    graph->mutable_node_features()->emplace(node_id, features_proto);
  }

  for (const auto& [node_id, neighbors] : _node_id_to_neighbors) {
    proto::data::GraphNeighbors neighbors_proto;
    *neighbors_proto.mutable_neighbors() = {neighbors.begin(), neighbors.end()};

    graph->mutable_neighbors()->emplace(node_id, neighbors_proto);
  }

  return graph;
}

template void GraphInfo::serialize(cereal::BinaryInputArchive&);
template void GraphInfo::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void GraphInfo::serialize(Archive& archive) {
  archive(_feature_dim, _node_id_to_feature_vector, _node_id_to_neighbors);
}

}  // namespace thirdai::automl::data