#include "UnigramVectorizer.h"

namespace thirdai::utils {

// UnigramVectorizer::~UnigramVectorizer() {}

UnigramVectorizer::UnigramVectorizer(uint32_t start_idx, uint32_t max_dim,
                                     VALUE_TYPE value_type)
    : StringVectorizer(start_idx, max_dim) {
  _value_type = value_type;
}

void UnigramVectorizer::vectorize(const std::string& str,
                                  std::vector<uint32_t>& indices,
                                  std::vector<float>& values) {
  std::unordered_map<uint32_t, float> emptyMap;
  vectorize(str, indices, values, emptyMap);
}

void UnigramVectorizer::UnigramVectorizer::vectorize(
    const std::string& str, std::vector<uint32_t>& indices,
    std::vector<float>& values,
    const std::unordered_map<uint32_t, float>& idfMap) {
  std::stringstream stream(str);
  std::string temp;
  std::unordered_map<uint32_t, float> ids;
  while (stream >> temp) {
    // Mapping to count frequencies of unique token ids.
    const char* converted = temp.c_str();
    u_int32_t len = temp.length();
    u_int32_t hash =
        MurmurHash(converted, len, _murmur_seed) % _max_dim + _start_idx;
    switch (_value_type) {
      case VALUE_TYPE::TFIDF:
        if (const float idf = idfMap.at(hash) != 0.0) {
          ids[hash] += idf;
        } else {
          ids[hash]++;
        }
        break;

      case VALUE_TYPE::BINARY:
        ids[hash] = 1;
        break;

      case VALUE_TYPE::FREQUENCY:
        ids[hash]++;
        break;

      default:
        break;
    }
    // TODO (henry): What about markers?
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
}

// void UnigramVectorizer::setGlobalFreq(GlobalFreq* global_freq) {
//       _global_freq = global_freq;
// }

}  // namespace thirdai::utils