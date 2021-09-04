#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "Dataset.h"
#include "MurmurHash3.h"

namespace thirdai::utils {

  enum class STRING_FEATURE_TYPE { N_GRAM };

  class StringDataset: public Dataset {

    public:

    // n-gram constructor
    StringDataset(uint64_t target_batch_size, uint64_t target_batch_num_per_load, STRING_FEATURE_TYPE type, uint32_t n)
    : Dataset(target_batch_size, target_batch_num_per_load) {
      assert(type == STRING_FEATURE_TYPE::N_GRAM);
    };

    virtual void loadNextBatchSet() {
      
    }

    private:
    std::unordered_map<uint32_t, float> _hashes;
    std::vector<uint32_t> _indices;
    std::vector<float> _values;

    void featurize_n_gram(const std::string& str, uint32_t n) {
      void *start = (void*) str.c_str();
      size_t len = str.length();
      _hashes.clear();
      for (size_t i = 0; i < len - n + 1; i++) {
        uint32_t hash;
        MurmurHash3_x86_32(start + i, n * sizeof(char), 341, (void *) &hash);
        _hashes[hash]++;
      }
      size_t new_cap = (_indices.size() + _hashes.size());
      _indices.reserve(new_cap * sizeof(uint32_t));
      _values.reserve(new_cap * sizeof(float));
      for (auto kv: _hashes) {
        _indices.push_back(kv.first);
        _values.push_back(kv.second);
      }
    }

    // TODO: Benchmark and test
  };
}