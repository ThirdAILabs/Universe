#pragma once

#include "NumstringEncodingInterface.h"
#include <dataset/src/utils/Conversions.h>

namespace thirdai::dataset {

/**
 * Interface for numstring encoding models.
 */
struct IndexValuePairs : public NumstringEncoding {

  explicit IndexValuePairs(uint32_t dim, char delim=':'): _dim(dim), _delim(delim) {}

  /**
   * Encodes a numstring as vector features.
   * This method may update the offset parameter.
   */
  void encodeNumstring(std::string_view numstr, BuilderVector& shared_feature_vector,
                       uint32_t& offset) final {
    size_t delim_pos = numstr.find(_delim);
    const std::string_view s(numstr.substr(0, delim_pos));
    uint32_t index = getNumberU32(s);
    float value;

    shared_feature_vector.addSingleFeature(offset + index, offset + value);
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
  char _delim;
};

}  // namespace thirdai::dataset