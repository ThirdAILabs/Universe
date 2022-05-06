#pragma once

#include <dataset/src/BuilderVectors.h>

namespace thirdai::dataset {

/**
 * Interface for text embedding models.
 */
struct TextEmbeddingModel {
  /**
   * Tokenizes each string in text, embeds these tokens, 
   * and composes the shared feature vector with these embeddings.
   */
  virtual void embedText(const std::string& text, BuilderVector& shared_feature_vector, uint32_t offset) = 0;

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