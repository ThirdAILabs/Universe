#include "UnigramVectorizer.h"

namespace thirdai::utils {

//UnigramVectorizer::~UnigramVectorizer() {}

UnigramVectorizer::UnigramVectorizer(uint32_t start_idx, uint32_t max_dim, VECTOR_TYPE vector_type) : StringVectorizer(start_idx, max_dim){
  _vector_type = vector_type;
}

void UnigramVectorizer::vectorize(const std::string& str,
                                          std::vector<uint32_t>& indices,
                                          std::vector<float>& values) {
    std::stringstream stream(str);
    std::string temp;
    std::unordered_map<uint32_t, float> ids;
    while (stream >> temp) {
      // Mapping to count frequencies of unique token ids.
      const char *converted = temp.c_str();
      u_int32_t len = temp.length();
      u_int32_t hash = MurmurHash(converted, len, _murmur_seed) % _max_dim;
      if (_vector_type == VECTOR_TYPE::TFIDF) {
        //int idf = _globalFreq->getIdf(temp);
        int idf = 1; // Placeholder
        ids[hash] += idf;
      }
      else {
        ids[hash] += 1;
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

}  // namespace thirdai::utils