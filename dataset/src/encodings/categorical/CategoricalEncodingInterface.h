#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>

namespace thirdai::dataset {

/**
 * Interface for categorical feature encoders.
 */
struct CategoricalEncoding {
  /**
   * Conceptually, encodes an categorical feature represented by an ID
   * as a vector and adds a segment containing this encoding to the vector.
   */
  virtual void encodeCategory(const std::string& id, SegmentedFeatureVector& vec) = 0;

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