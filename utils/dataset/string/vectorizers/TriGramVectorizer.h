#pragma once
#include "../../../hashing/MurmurHash.h"
#include "StringVectorizer.h"
#include <tgmath.h>
#include <iostream>

namespace thirdai::utils {
class TriGramVectorizer : public StringVectorizer {
 public:
  TriGramVectorizer();

  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values);

 private:
  uint8_t *_hashC;
  uint16_t _37x[37];
  uint16_t _37x37x[37];
};
}  // namespace thirdai::utils