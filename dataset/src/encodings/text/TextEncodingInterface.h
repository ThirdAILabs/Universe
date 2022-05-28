#pragma once

#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Interface for text encoders.
 */
struct TextEncoding {
  /**
   * Conceptually, encodes a string as a vector and extends
   * the given vector with this encoding.
   */
  virtual void encodeText(const std::string& text, SegmentedFeatureVector& vec) = 0;

  /**
   * True if the encoder produces dense features, False otherwise.
   */
  virtual bool isDense() = 0;

  /**
   * The dimension of the encoding.
   */
  virtual uint32_t featureDim() = 0;
};

}  // namespace thirdai::dataset