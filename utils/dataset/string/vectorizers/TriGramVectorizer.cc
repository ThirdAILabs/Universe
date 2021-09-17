#include "TriGramVectorizer.h"

namespace thirdai::utils {
TriGramVectorizer::TriGramVectorizer(GlobalFreq* globalFreq): StringVectorizer(globalFreq) {
  _dim = 50653;  // 37 ^ 3; 26 letters in the alphabet, 10 numbers, and space.
                 // 3 characters per token.
                 
  // Make minimum perfect hash.
  _hashC = new uint8_t[256];
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
    const uint8_t *char_int_ptr = reinterpret_cast<const uint8_t *>(start + i);
    hash += _hashC[char_int_ptr[0]] << 6;
    hash += _37x[_hashC[char_int_ptr[1]]];
    hash += _37x37x[_hashC[char_int_ptr[2]]];
    ids[hash] += _globalFreq->getIdf(1);
  }
  // Resize the vector to the number of unique token IDs.
  indices.resize(ids.size());
  values.resize(ids.size());
  size_t i = 0;
  // This overwrites the previous contents of indices and values.
  for (auto kv : ids) {
    indices[i] = kv.first;
    values[i] = kv.second;
    i++;
  }
};
}  // namespace thirdai::utils