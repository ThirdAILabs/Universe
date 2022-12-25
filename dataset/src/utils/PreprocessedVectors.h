#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

struct PreprocessedVectors {
  PreprocessedVectors(std::unordered_map<std::string, BoltVector>&& vectors,
                      uint32_t dim)
      : vectors(std::move(vectors)), dim(dim) {}

  std::unordered_map<std::string, BoltVector> vectors;
  uint32_t dim;

  void appendPreprocessedFeaturesToVector(const std::string& key,
                                          SegmentedFeatureVector& vec) {
    if (!vectors.count(key)) {
      return;
    }

    const auto& vector = vectors.at(key);
    if (vector.isDense()) {
      for (uint32_t i = 0; i < vector.len; i++) {
        vec.addDenseFeatureToSegment(vector.activations[i]);
      }
    } else {
      for (uint32_t i = 0; i < vector.len; i++) {
        vec.addSparseFeatureToSegment(vector.neurons[i], vector.activations[i]);
      }
    }
  }

 private:
  // Private constructor for cereal
  PreprocessedVectors() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(vectors, dim);
  }
};

using PreprocessedVectorsPtr = std::shared_ptr<PreprocessedVectors>;

}  // namespace thirdai::dataset
