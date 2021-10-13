#pragma once
//#include "../StringDataset.h"
// #include "../GlobalFreq.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <exception>
// #include "../IDFMap.h"

namespace thirdai::utils {
enum class VALUE_TYPE { TFIDF, FREQUENCY, BINARY };

/**
 * Interface for extracting sparse vector indices and values out of strings.
 * The constructor of the derived class has to set _dim.
 */
class StringVectorizer {
 public:
  /**
   * start_idx: The smallest possible non-zero index of the produced vector.
   * The non-zero indices produced by the vectorizer should be shifted by
   * + _start_idx.
   *
   * max_dim: The maximum dimension of the produced vector.
   * max_dim cannot be 0.
   */
  StringVectorizer(uint32_t start_idx, uint32_t max_dim)
      : _start_idx(start_idx), _max_dim(max_dim) {
    constructorHelper();
  };

  void constructorHelper() const {
    if (_max_dim < 1) {
      throw std::invalid_argument("String vectorizer does not accept max_dim < 1");
    }
  }

  /**
   * Returns the dimension of the vector.
   */
  uint64_t getDimension() const { return _dim; };

  /**
   * Takes in a string 'str' and fills out 'indices' and 'values' vectors,
   * corresponding with the indices and values arrays of a sparse vector.
   * 'indices' and 'values' are not necessarily empty. This method appends
   * to 'indices' and 'values'.
   * 
   * TODO(geordie): document the differnce between the signature that takes in emptyMap and the one that doesn't
   */
  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values) {
    std::unordered_map<uint32_t, float> emptyMap;
    vectorize(str, indices, values, emptyMap);
  };

  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values, const std::unordered_map<uint32_t, float>& idfMap) = 0;

  virtual ~StringVectorizer() {}

 protected:
  /**
   * The smallest possible non-zero index of the produced vector.
   * The non-zero indices produced by the vectorizer should be shifted by
   * + _start_idx.
   */
  uint32_t _start_idx;

  /**
   * The maximum dimension of the produced vector.
   * _max_dim cannot be 0.
   */
  uint32_t _max_dim;

  /**
   * Dimension of the vector.
   */
  uint32_t _dim;

};
}  // namespace thirdai::utils