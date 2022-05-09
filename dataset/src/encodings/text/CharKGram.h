#pragma once

#include <algorithm>
#include <limits>
#include "TextEncodingInterface.h"

namespace thirdai::dataset {

struct CharKGram: public TextEncoding {
  
  CharKGram(uint32_t k, uint32_t dim): _k(k), _dim(dim) {}

  void embedText(const std::string& text, BuilderVector& shared_feature_vector, uint32_t idx_offset) final {
    std::vector<uint32_t> indices;

    if (text.size() < _k) {
      return;
    }
    
    int64_t power = 1;
    for (size_t i = 0; i < _k; i++) {
      power = (power * PRIME_BASE) % PRIME_MOD;
    }

    int64_t hash = 0;
    for (size_t i = 0; i < text.size(); i++) {
      // Add last letter
      hash = hash * PRIME_BASE + text[i];
      hash %= PRIME_MOD;

      // Remove first character if needed
      if (i >= _k) {
        hash -= power * text[i - _k] % PRIME_MOD;
        if (hash < 0) {
          hash += PRIME_MOD;
        }
      }

      if (i >= _k - 1) {
        indices.push_back(hash % _dim + idx_offset);
      }
    }

    incrementAtIndices(shared_feature_vector, indices, 1.0);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:

  static void incrementAtIndices(BuilderVector& shared_feature_vector, std::vector<uint32_t>& indices, float inc) {
    std::sort(indices.begin(), indices.end());
    uint32_t impossible = std::numeric_limits<uint32_t>::max(); // Way greater than prime mod so no index will be equal to this.
    indices.push_back(impossible);
    uint32_t last_idx = impossible;
    float last_idx_val = 0.0;
    for (uint32_t idx : indices) {
      if (idx != last_idx && last_idx != impossible) {
        shared_feature_vector.addSingleFeature(idx, last_idx_val);
        last_idx_val = 0.0;
      }
      last_idx = idx;
      last_idx_val += inc;
    }
  }
  
  static constexpr uint32_t PRIME_BASE = 257;
  static constexpr uint32_t PRIME_MOD = 1000000007;
  
  uint32_t _k;
  uint32_t _dim;
};

} // namespace thirdai::dataset