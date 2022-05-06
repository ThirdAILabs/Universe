#pragma once

#include "CategoricalEmbeddingModelInterface.h"

namespace thirdai::dataset {

struct OneHotEncoding : public CategoricalEmbeddingModel {
  explicit OneHotEncoding(uint32_t dim): _dim(dim) {}

  /**
   * Maps an id to an embedding
   */
  void embedCategory(uint32_t id, BuilderVector& shared_feature_vector, uint32_t offset) final {
    shared_feature_vector.addSingleFeature(id % _dim + offset, 1.0);
  };

  /**
   * True if the model produces dense features, False otherwise.
   */
  bool isDense() final {
    return false;
  };

  /**
   * The dimension of the embedding produced by this model.
   */
  uint32_t featureDim() final {
    return _dim;
  };

 private:
  uint32_t _dim;
};

} // namespace thirdai::dataset