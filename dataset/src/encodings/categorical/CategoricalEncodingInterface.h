#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/ExtendableVectors.h>

namespace thirdai::dataset {

/**
 * Interface for categorical feature encoders.
 */
class CategoricalEncoding {
 public:
  /**
   * Conceptually, encodes an categorical feature represented by an ID
   * as a vector and extends the given vector with this encoding.
   */
  virtual void encodeCategory(const std::string& id, ExtendableVector& vec) = 0;

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