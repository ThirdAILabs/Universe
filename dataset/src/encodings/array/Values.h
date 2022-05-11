#pragma once

#include "ArrayEncodingInterface.h"
#include <stdexcept>

namespace thirdai::dataset {

/**
 * Interface for numstring encoding models.
 */
struct Values : public ArrayEncoding {

  explicit Values(uint32_t dim): _dim(dim) {}

  /**
   * Encodes a numstring as vector features.
   * This method may update the offset parameter.
   */
  void encodeArray(const std::function<std::optional<std::string>()>& next_elem, 
                   BuilderVector& shared_feature_vector,
                   uint32_t offset) final {
    
    uint32_t elems_added = 0;
    std::optional<std::string> elem;
    while ((elem = next_elem()).has_value()) {
      const auto& numstr = elem.value();
      
      char* end;
      float value = std::strtof(numstr.c_str(), &end);
      shared_feature_vector.addSingleFeature(offset + elems_added, value);

      elems_added++;
      
      if (elems_added >= _dim) {
        std::stringstream ss;
        ss << "[Values] Given dim = " << _dim << " but given " << elems_added << " elements.";
        throw std::invalid_argument(ss.str());
      }
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