#include "MurmurHash.h"
#pragma once
#include "StringVectorizer.h"
#include <tgmath.h>

namespace thirdai::utils {
class TriGramVectorizer : public StringVectorizer {
 public:
  TriGramVectorizer(GlobalFreq* globalFreq) : StringVectorizer(globalFreq) {
    _dim = 50653;  // 37 ^ 3; 26 letters in the alphabet, 10 numbers, and space.
                   // 3 characters per token.
  };

  virtual void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                         std::vector<float>& values, VECTOR_TYPE vector_type) {
    const char* start = str.c_str();
    size_t len = str.length();
    // Mapping to count frequencies of unique token ids.
    std::unordered_map<uint32_t, float> ids;
    switch (vector_type) {
      case VECTOR_TYPE::MURMUR:
        /* code */
        for (size_t i = 0; i < len - 2; i++) {
          // Using MurmurHash instead of a rolling hash because 3 characters
          // take up less than 32 bits, so this is only one iteration in
          // MurmurHash, and it has very good distribution.
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
        break;
      case VECTOR_TYPE::TFIDF:
        indices.clear();
        values.clear();
        for (size_t i = 0; i < len - 2; i++) {
          std::string token = str.substr(i, i + 2);
          // Use a default idf for trigram tokens
          int default_idf = 1;
          int tf = _globalFreq->getTF(token, str);
          int tokenID = _globalFreq->getTokenID(token);
          indices.push_back(tokenID);
          values.push_back(default_idf * tf);
          // TODO: What about markers?
        }
        break;

      default:
        break;
    }
  };
};
}  // namespace thirdai::utils