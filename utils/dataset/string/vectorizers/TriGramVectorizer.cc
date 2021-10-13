#include "TriGramVectorizer.h"

namespace thirdai::utils {

TriGramVectorizer::TriGramVectorizer(uint32_t start_idx, uint32_t max_dim,
                                     VALUE_TYPE value_type)
    : StringVectorizer(start_idx, max_dim), _value_type(value_type) {
  constructorHelper();
}

void TriGramVectorizer::constructorHelper() {
  uint32_t original_dim = 50653;  // 37 ^ 3; 26 letters in the alphabet, 10
                                  // numbers, and space. 3 characters per token.
  _dim = std::min(original_dim, _max_dim);

  // Make minimum perfect hash.
  _hashC = new uint8_t[256]();  // End with () to value-initialize to 0 so it
                                // does not break when given random symbols.
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
}

void TriGramVectorizer::vectorize(const std::string& str,
                                  std::vector<uint32_t>& indices,
                                  std::vector<float>& values) {
  std::unordered_map<uint32_t, float> emptyMap;
  vectorize(str, indices, values, emptyMap);
}

void TriGramVectorizer::vectorize(
    const std::string& str, std::vector<uint32_t>& indices,
    std::vector<float>& values,
    const std::unordered_map<uint32_t, float>& idfMap) {
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
    hash = (hash % _dim) + _start_idx;
    switch (_value_type) {
      case VALUE_TYPE::FREQUENCY:
        ids[hash]++;
        break;
      case VALUE_TYPE::TFIDF:
        if (idfMap.count(hash) != 0) {
          ids[hash] += idfMap.at(hash);
        } else {
          ids[hash]++;
        }
        break;
      case VALUE_TYPE::BINARY:
        ids[hash] = 1;
        break;
      default:
        break;
    }
  }
  // Reserve space for new unique token IDs.
  indices.reserve(indices.size() + ids.size());
  values.reserve(values.size() + ids.size());
  for (auto kv : ids) {
    indices.push_back(kv.first);
    values.push_back(kv.second);
  }
}

}  // namespace thirdai::utils
