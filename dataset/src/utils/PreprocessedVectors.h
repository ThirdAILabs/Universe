#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

struct PreprocessedVectors {
  PreprocessedVectors(std::vector<BoltVector>&& vectors, uint32_t dim)
      : vectors(std::move(vectors)), dim(dim) {}

  std::vector<BoltVector> vectors;
  uint32_t dim;

  void appendPreprocessedFeaturesToVector(uint32_t id,
                                          SegmentedFeatureVector& vec) {
    if (id >= vectors.size()) {
      throw std::invalid_argument("Invalid preprocessed vector ID " +
                                  std::to_string(id) + ". There are only " +
                                  std::to_string(vectors.size()) + " vectors.");
    }
    auto vector = vectors.at(id);
    if (vector.isDense()) {
      for (uint32_t i = 0; i < vector.len; i++) {
        vec.addDenseFeatureToSegment(vector.activations[i]);
      }
    } else {
      for (uint32_t i = 0; i < vector.len; i++) {
        vec.addSparseFeatureToSegment(vector.active_neurons[i],
                                      vector.activations[i]);
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