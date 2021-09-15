#include "TriGramVectorizer.h"

namespace thirdai::utils {
TriGramVectorizer::TriGramVectorizer() {
  _dim = 50653;  // 37 ^ 3; 26 letters in the alphabet, 10 numbers, and space.
                 // 3 characters per token.
};

void TriGramVectorizer::vectorize(const std::string& str,
                                  std::vector<uint32_t>& indices,
                                  std::vector<float>& values) {
  const char* start = str.c_str();
  size_t len = str.length();
  // Mapping to count frequencies of unique token ids.
  std::unordered_map<uint32_t, float> ids;

  for (size_t i = 0; i < len - 2; i++) {
    // Using MurmurHash instead of a rolling hash because 3 characters take up
    // less than 32 bits, so this is only one iteration in MurmurHash, and it
    // has very good distribution.
    uint32_t hash = MurmurHash(start + i, 3 * sizeof(char), 341) % _dim;
    ids[hash]++;
  }
  // Resize the vector to the number of unique token IDs.
  indices.resize(ids.size() * sizeof(uint32_t));
  values.resize(ids.size() * sizeof(float));
  size_t i = 0;
  // This overwrites the previous contents of indices and values.
  for (auto kv : ids) {
    indices[i] = kv.first;
    values[i] = kv.second;
    i++;
  }
};
}  // namespace thirdai::utils