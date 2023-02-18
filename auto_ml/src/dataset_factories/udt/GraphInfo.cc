#include "GraphInfo.h"

namespace thirdai::automl::data {

const std::vector<float>& GraphInfo::featureVector(uint64_t node_id) {
  if (!_node_id_to_feature_vector.count(node_id)) {
    throw std::runtime_error("No feature vector currently stored for node " +
                             std::to_string(node_id));
  }

  return _node_id_to_feature_vector.at(node_id);
}

std::vector<uint64_t>& GraphInfo::neighbors(uint64_t node_id) {
  if (!_node_id_to_neighbors.count(node_id)) {
    throw std::runtime_error("No neighbors vector currently stored for node " +
                             std::to_string(node_id));
  }

  return _node_id_to_neighbors.at(node_id);
}

void GraphInfo::insertNode(uint64_t node_id, std::vector<float> features,
                           std::vector<uint64_t> neighbors) {
  _node_id_to_feature_vector.at(node_id) = std::move(features);
  _node_id_to_neighbors.at(node_id) = std::move(neighbors);
}

void GraphInfo::clear() {
  _node_id_to_feature_vector.clear();
  _node_id_to_neighbors.clear();
}

}  // namespace thirdai::automl::data