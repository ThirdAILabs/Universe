#pragma once
#include "../../../hashing/MurmurHash.h"
#include "StringVectorizer.h"
#include <tgmath.h>

namespace thirdai::utils {
class TriGramVectorizer : public StringVectorizer {
 public:
  TriGramVectorizer();

  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values);
};
}  // namespace thirdai::utils