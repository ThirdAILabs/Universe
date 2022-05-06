#pragma once

#include <dataset/src/BuilderVectors.h>

namespace thirdai::dataset {

/**
 * Interface for text embedding models.
 */
struct CategoricalEmbeddingModel {
  /**
   * Maps an id to an embedding
   */
  virtual void embedCategory(uint32_t id, BuilderVector& shared_feature_vector, uint32_t offset) = 0;

  /**
   * True if the model produces dense features, False otherwise.
   */
  virtual bool isDense() = 0;

  /**
   * The dimension of the embedding produced by this model.
   */
  virtual uint32_t featureDim() = 0;
};

} // namespace thirdai::dataset