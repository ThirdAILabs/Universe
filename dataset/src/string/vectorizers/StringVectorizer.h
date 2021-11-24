#pragma once

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {
enum class StringVectorizerValue { TFIDF, FREQUENCY, BINARY };

/**
 * Interface for extracting sparse vector indices and values out of strings.
 * The constructor of the derived class has to set _dim.
 */
class StringVectorizer {
 public:
  /**
   * start_idx: The smallest possible non-zero index of the produced vector.
   * This is used because vectors are concatenated with other vectors.
   * The non-zero indices produced by the vectorizer should be shifted by
   * + _start_idx.
   *
   * max_dim: The maximum dimension of the produced vector.
   * max_dim cannot be 0.
   */
  StringVectorizer(uint32_t start_idx, uint32_t max_dim,
                   StringVectorizerValue value_type)
      : _start_idx(start_idx), _max_dim(max_dim), _value_type(value_type) {
    constructorHelper();
  };

  void constructorHelper() const {
    if (_max_dim < 1) {
      throw std::invalid_argument(
          "String vectorizer does not accept max_dim < 1, max_dim = " +
          std::to_string(_max_dim));
    }
  }

  /**
   * Returns the dimension of the vector.
   */
  uint64_t getDimension() const { return _dim; };

  /**
   * Takes in a string 'str' and fills out 'indexToValueMap' which maps
   * the indices to the values of a sparse vector.
   * indexToValueMap are not necessarily empty. This method adds new
   * entries to indexToValueMap.
   * However, the keys added by this method must not overlap with the
   * existing keys in the map. If it does, then it can be overwritten.
   */
  virtual void fillIndexToValueMap(
      const std::string& str,
      std::unordered_map<uint32_t, float>& index_to_value_map,
      const std::unordered_map<uint32_t, float>& idf_map) = 0;

  /**
   * Helper function to set the value of an index_to_value_map based
   * on _value_type. Avoids repeating the same code in all string
   * vectorizer derived classes.
   */
  void setMapValue(std::unordered_map<uint32_t, float>& index_to_value_map,
                   uint32_t key,
                   const std::unordered_map<uint32_t, float>& idf_map) {
    switch (_value_type) {
      case StringVectorizerValue::BINARY:
        index_to_value_map[key] = 1;
        break;
      case StringVectorizerValue::FREQUENCY:
        index_to_value_map[key] += 1.0;
        break;
      case StringVectorizerValue::TFIDF:
        if (idf_map.find(key) != idf_map.end()) {
          index_to_value_map[key] += idf_map.at(key);
        } else {
          index_to_value_map[key] += 1.0;
        }
      default:
        break;
    }
  }

  virtual ~StringVectorizer() {}

 protected:
  /**
   * The smallest possible non-zero index of the produced vector.
   * This is used because vectors are concatenated with other vectors.
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

  /**
   * Whether the vector values are binary, frequency, or tf-idf.
   */
  StringVectorizerValue _value_type;
};
}  // namespace thirdai::dataset