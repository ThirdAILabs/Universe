#pragma once

#include "NumstringEncodingInterface.h"
#include <stdexcept>

namespace thirdai::dataset {

/**
 * Interface for numstring encoding models.
 */
struct Indices : public NumstringEncoding {

  explicit Indices(uint32_t dim): _dim(dim) {}

  /**
   * Encodes a numstring as vector features.
   * This method may update the offset parameter.
   */
  void encodeNumstring(const std::string& numstr, BuilderVector& shared_feature_vector,
                       uint32_t& offset) final {

    char* end;
    uint32_t index = std::strtoul(numstr.c_str(), &end, 10);
    
    if (index >= _dim) {
      std::stringstream ss;
      ss << "[Indices] Given dim = " << _dim << " but got index = " << index;
      throw std::invalid_argument(ss.str());
    }

    shared_feature_vector.addSingleFeature(offset + index, 1.0);
  }

  /**
   * True if the model produces dense features, False otherwise.
   */
  bool isDense() final {
    return false;
  }

  /**
   * The dimension of the encoding produced by this model.
   */
  uint32_t featureDim() final {
    return _dim;
  }

 private:
  uint32_t _dim;
};

}  // namespace thirdai::dataset