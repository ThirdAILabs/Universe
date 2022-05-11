#pragma once

#include "ArrayEncodingInterface.h"
#include <stdexcept>

namespace thirdai::dataset {

/**
 * Interface for numstring encoding models.
 */
struct Indices : public ArrayEncoding {

  explicit Indices(uint32_t dim): _dim(dim) {}

  /**
   * Encodes an array iterable as vector features.
   * This method may update the offset parameter.
   */
  void encodeArray(const std::function<std::optional<std::string>()>& next_elem, 
                   BuilderVector& shared_feature_vector,
                   uint32_t& offset) final {
    std::optional<std::string> elem;
    while ((elem = next_elem()).has_value()) {
      const auto& numstr = elem.value();
      
      char* end;
      uint32_t index = std::strtoul(numstr.c_str(), &end, 10);
      
      if (index >= _dim) {
        std::stringstream ss;
        ss << "[Indices] Given dim = " << _dim << " but got index = " << index;
        throw std::invalid_argument(ss.str());
      }

      shared_feature_vector.addSingleFeature(offset + index, 1.0);
    }
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