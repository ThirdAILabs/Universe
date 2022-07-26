#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>

namespace thirdai::dataset {

/**
 * Interface for categorical feature encoders.
 */
class CategoricalEncoding {
 public:
  /**
   * Conceptually, encodes an categorical feature represented by an ID
   * as a vector and adds a segment containing this encoding to the vector.
   */
  virtual std::exception_ptr encodeCategory(std::string_view id, SegmentedFeatureVector& vec,
                              uint32_t offset) = 0;

  /**
   * True if the encoder produces dense features, False otherwise.
   */
  virtual bool isDense() const = 0;

  /**
   * The dimension of the encoding.
   */
  virtual uint32_t featureDim() const = 0;
};

}  // namespace thirdai::dataset