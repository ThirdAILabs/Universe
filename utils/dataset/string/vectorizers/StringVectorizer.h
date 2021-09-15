#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::utils {
/**
 * Interface for extracting sparse vector indices and values out of strings.
 * The constructor of the derived class has to set _dim.
 */
class StringVectorizer {
 public:
  /**
   * Returns the dimension of the vector.
   */
  uint64_t getDimension() const { return _dim; };

  /**
   * Takes in a string 'str' and fills out 'indices' and 'values' vectors,
   * corresponding with the indices and values arrays of a sparse vector.
   * 'indices' and 'values' are not necessarily empty so this method has
   * to ensure that 'indices' and 'values' are overwritten.
   */
  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values) = 0;

 protected:
  /**
   * Dimension of the vector.
   */
  uint32_t _dim;
};
}  // namespace thirdai::utils