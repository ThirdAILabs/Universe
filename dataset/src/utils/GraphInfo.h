#pragma once

#include <cereal/access.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::automl {

class GraphConstructionError : public std::runtime_error {
 public:
  explicit GraphConstructionError(const std::string& message)
      : std::runtime_error(
            "The model's stored graph is in an unexpected state: " + message){};
};

class GraphInfo {
 public:
  explicit GraphInfo(uint64_t feature_dim) : _feature_dim(feature_dim) {}

  void clear();

  const std::vector<float>& featureVector(uint64_t node_id) const;

  const std::vector<uint64_t>& neighbors(uint64_t node_id) const;

  void insertNode(uint64_t node_id, std::vector<float> features,
                  std::vector<uint64_t> neighbors);

  uint64_t featureDim() const { return _feature_dim; }

 private:
  uint64_t _feature_dim;
  std::unordered_map<uint64_t, std::vector<float>> _node_id_to_feature_vector;
  std::unordered_map<uint64_t, std::vector<uint64_t>> _node_id_to_neighbors;

  GraphInfo() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

using GraphInfoPtr = std::shared_ptr<GraphInfo>;

}  // namespace thirdai::automl