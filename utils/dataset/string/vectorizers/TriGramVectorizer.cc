#include "TriGramVectorizer.h"

namespace thirdai::utils {

TriGramVectorizer::TriGramVectorizer(uint32_t start_idx, uint32_t max_dim)
    : StringVectorizer(start_idx, max_dim) {
  uint32_t original_dim = 50653;  // 37 ^ 3; 26 letters in the alphabet, 10
                                  // numbers, and space. 3 characters per token.
  _dim = std::min(original_dim, _max_dim);

  // Make minimum perfect hash.
  _hashC = new uint8_t[128];  // Among lower case letters, numbers and space,
                              // the highest ascii value is 122.
  // Space
  _hashC[32] = 0;
  // Numbers
  for (size_t i = 0; i < 10; i++) {
    _hashC[48 + i] = 1 + i;
  }
  // Lower case letters
  for (size_t i = 0; i < 26; i++) {
    _hashC[97 + i] = 11 + i;
  }
  // Precompute calculations
  for (uint16_t i = 0; i < 37; i++) {
    _37x[i] = i * 37;
    _37x37x[i] = _37x[i] * 37;
  }
};

void TriGramVectorizer::vectorize(const std::string& str,
                                  std::vector<uint32_t>& indices,
                                  std::vector<float>& values) {
  const char* start = str.c_str();
  size_t len = str.length();
  // Mapping to count frequencies of unique token ids.
  std::unordered_map<uint32_t, float> ids;
  for (size_t i = 0; i < len - 2; i++) {
    // Can make rolling but it would only cut down one operation.
    uint32_t hash = 0;
    const uint8_t* char_int_ptr = reinterpret_cast<const uint8_t*>(start + i);
    hash += _hashC[char_int_ptr[0]];
    hash += _37x[_hashC[char_int_ptr[1]]];
    hash += _37x37x[_hashC[char_int_ptr[2]]];
    ids[(hash % _dim) + _start_idx]++;
  }
  // Reserve space for new unique token IDs.
  indices.reserve(indices.size() + ids.size());
  values.reserve(values.size() + ids.size());
  for (auto kv : ids) {
    indices.push_back(kv.first);
    values.push_back(kv.second);
  }
};
}  // namespace thirdai::utils