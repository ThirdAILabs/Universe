#include "GraphInfo.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <utility>

namespace thirdai::automl {

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

template void GraphInfo::serialize(cereal::BinaryInputArchive&);
template void GraphInfo::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void GraphInfo::serialize(Archive& archive) {
  archive(_feature_dim, _node_id_to_feature_vector, _node_id_to_neighbors);
}

}  // namespace thirdai::automl