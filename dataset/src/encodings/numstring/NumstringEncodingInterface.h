#pragma once

#include <dataset/src/utils/BuilderVectors.h>

namespace thirdai::dataset {

/**
 * Interface for numstring encoding models.
 */
struct NumstringEncoding {
  /**
   * Encodes a numstring as vector features.
   * This method may update the offset parameter.
   */
  virtual void encodeNumstring(const std::string& numstr, BuilderVector& shared_feature_vector,
                               uint32_t& offset) = 0;

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