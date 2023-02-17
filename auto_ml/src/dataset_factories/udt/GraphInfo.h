#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// TODO(Josh): Fix file location
namespace thirdai::automl::data {

class GraphInfo {
 public:
  explicit GraphInfo(uint64_t feature_dim) : _feature_dim(feature_dim) {}

  void clear();

  const std::vector<float>& featureVector(uint64_t node_id);

  std::vector<uint64_t>& neighbors(uint64_t node_id);

  void insertNode(uint64_t node_id, std::vector<float> features,
                  std::vector<uint64_t> neighbors);

  uint64_t featureDim();

 private:
  uint64_t _feature_dim;
  // TODO(Josh): Consider replacing with Eigen
  std::unordered_map<uint64_t, std::vector<float>> _node_id_to_feature_vector;
  std::unordered_map<uint64_t, std::vector<uint64_t>> _node_id_to_neighbors;
};

using GraphInfoPtr = std::shared_ptr<GraphInfo>;

}  // namespace thirdai::automl::data