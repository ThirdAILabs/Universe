#include "StringTokenizer.h"
#include "MurmurHash.h"

namespace thirdai::utils {
  class NGramTokenizer : public StringTokenizer {
    public:
    NGramTokenizer(uint32_t n): _n(n) {};
    virtual void tokenize(const std::string& str, std::unordered_map<uint32_t, float>& hashes, std::vector<uint32_t>& indices, std::vector<float>& values) {
      const char *start = str.c_str();
      size_t len = str.length();
      hashes.clear();
      for (size_t i = 0; i < len - _n + 1; i++) {
        uint32_t hash = MurmurHash(start + i, _n * sizeof(char), 341);
        hashes[hash]++;
      }
      indices.resize(hashes.size() * sizeof(uint32_t));
      values.resize(hashes.size() * sizeof(float));
      size_t i = 0;
      for (auto kv: hashes) {
        indices[i] = kv.first;
        values[i] = kv.second;
        i++;
      }
    };
    private:
    uint32_t _n;
  };
}