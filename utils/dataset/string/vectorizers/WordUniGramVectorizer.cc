#include "WordUniGramVectorizer.h"

namespace thirdai::utils::dataset {

WordUniGramVectorizer::WordUniGramVectorizer(uint32_t start_idx,
                                             uint32_t max_dim,
                                             StringVectorizerValue value_type)
    : StringVectorizer(start_idx, max_dim, value_type) {}

void WordUniGramVectorizer::fillIndexToValueMap(
    const std::string& str,
    std::unordered_map<uint32_t, float>& index_to_value_map,
    const std::unordered_map<uint32_t, float>& idf_map) {
  std::stringstream stream(str);
  std::string temp;
  std::unordered_map<uint32_t, float> ids;
  while (stream >> temp) {
    // Mapping to count frequencies of unique token ids.
    const char* converted = temp.c_str();
    u_int32_t len = temp.length();
    u_int32_t hash =
        MurmurHash(converted, len, _murmur_seed) % _max_dim + _start_idx;
    setMapValue(index_to_value_map, hash, idf_map);
  }
}

}  // namespace thirdai::utils::dataset