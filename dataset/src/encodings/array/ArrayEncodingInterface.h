#pragma once

#include <dataset/src/utils/BuilderVectors.h>
#include <optional>

namespace thirdai::dataset {

/**
 * Interface for numstring encoding models.
 */
struct ArrayEncoding {
  /**
   * Encodes a numstring as vector features.
   * This method may update the offset parameter.
   */
  virtual void encodeArray(const std::function<std::optional<std::string>()>& next_elem, 
                           BuilderVector& shared_feature_vector,
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