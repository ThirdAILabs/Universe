#pragma once
#include "StringVectorizer.h"
#include <ctgmath>
#include <iostream>

namespace thirdai::utils::dataset {
class CharTriGramVectorizer : public StringVectorizer {
 public:
  CharTriGramVectorizer(uint32_t start_idx, uint32_t max_dim,
                        StringVectorizerValue value_type);

  void fillIndexToValueMap(
      const std::string& str,
      std::unordered_map<uint32_t, float>& index_to_value_map,
      const std::unordered_map<uint32_t, float>& idf_map) override;

  ~CharTriGramVectorizer() {}

 private:
  /** A vector mapping characters to a number between 0 and 36 inclusive.
   * Space -> 0
   * Numbers -> 1 ... 10
   * Letters (not case sensitive) -> 11 ... 36
   */

  std::vector<uint8_t> _character_hash;
};
}  // namespace thirdai::utils::dataset