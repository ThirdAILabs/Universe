#include "CharTriGramVectorizer.h"

namespace thirdai::utils::dataset {

CharTriGramVectorizer::CharTriGramVectorizer(uint32_t start_idx,
                                             uint32_t max_dim,
                                             StringVectorizerValue value_type)
    : StringVectorizer(start_idx, max_dim, value_type),
      _character_hash(std::vector<uint8_t>(256)) {
  uint32_t original_dim = 50653;  // 37 ^ 3; 26 letters in the alphabet, 10
                                  // numbers, and space. 3 characters per token.
  _dim = std::min(original_dim, _max_dim);

  // Make perfect hash for letters (not case sensitive), numbers, and space.
  // Space
  _character_hash[32] = 0;
  // Numbers
  for (size_t i = 0; i < 10; i++) {
    _character_hash[48 + i] = 1 + i;
  }
  // Upper case letters
  for (size_t i = 0; i < 26; i++) {
    _character_hash[65 + i] = 11 + i;
  }
  // Lower case letters
  for (size_t i = 0; i < 26; i++) {
    _character_hash[97 + i] = 11 + i;
  }
}

void CharTriGramVectorizer::fillIndexToValueMap(
    const std::string& str,
    std::unordered_map<uint32_t, float>& index_to_value_map,
    const std::unordered_map<uint32_t, float>& idf_map) {
  const char* start = str.c_str();
  size_t len = str.length();

  for (size_t i = 0; i < len - 2; i++) {
    // Calculate hash value.
    // Can make a rolling hash but the sequence is too short to be worth it.
    uint32_t hash = 0;
    const uint8_t* char_int_ptr = reinterpret_cast<const uint8_t*>(start + i);
    hash += _character_hash[char_int_ptr[0]];
    hash += 37 * _character_hash[char_int_ptr[1]];
    hash += 1369 * _character_hash[char_int_ptr[2]];  // 1369 is 37^2

    setMapValue(index_to_value_map, (hash % _dim) + _start_idx, idf_map);
  }
}
}  // namespace thirdai::utils::dataset