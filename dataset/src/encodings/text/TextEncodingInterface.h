#pragma once

#include <dataset/src/utils/BuilderVectors.h>

namespace thirdai::dataset {

/**
 * Interface for text encoding models.
 */
struct TextEncoding {
  /**
   * Tokenizes each string in text, embeds these tokens,
   * and composes the shared feature vector with these encodings.
   */
  virtual void encodeText(const std::string& text,
                         BuilderVector& shared_feature_vector,
                         uint32_t offset) = 0;

  /**
   * True if the model produces dense features, False otherwise.
   */
  virtual bool isDense() = 0;

  /**
   * The dimension of the encoding produced by this model.
   */
  virtual uint32_t featureDim() = 0;
};

}  // namespace thirdai::dataset