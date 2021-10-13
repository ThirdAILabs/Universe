#pragma once
#include "StringVectorizer.h"
#include <ctgmath>
#include <iostream>

namespace thirdai::utils {
class TriGramVectorizer : public StringVectorizer {
 public:
  TriGramVectorizer(uint32_t start_idx, uint32_t max_dim,
                    VALUE_TYPE value_type);

  void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                 std::vector<float>& values) override;

  void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                 std::vector<float>& values,
                 const std::unordered_map<uint32_t, float>& idfMap) override;

 private:
  // The following member variables are for storing the minimum perfect hash
  // for trigrams containing only lowercase letters, numbers, and spaces.
  uint8_t* _hashC;
  uint16_t _37x[37];
  uint16_t _37x37x[37];
  VALUE_TYPE _value_type;

  void constructorHelper();
};
}  // namespace thirdai::utils