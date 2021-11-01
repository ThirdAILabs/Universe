#pragma once
#include "../../../hashing/MurmurHash.h"
#include "StringVectorizer.h"
#include <sstream>

namespace thirdai::utils::dataset {

class WordUniGramVectorizer : public StringVectorizer {
 public:
  // UnigramVectorizer(GlobalFreq* globalFreq) : StringVectorizer(globalFreq){}
  WordUniGramVectorizer(uint32_t start_idx, uint32_t max_dim,
                        StringVectorizerValue value_type);

  void fillIndexToValueMap(
      const std::string& str,
      std::unordered_map<uint32_t, float>& index_to_value_map,
      const std::unordered_map<uint32_t, float>& idf_map) override;

 private:
  /* data */
  u_int32_t _murmur_seed = 42;
};

}  // namespace thirdai::utils::dataset