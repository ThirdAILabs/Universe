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

    shared_feature_vector.incrementAtIndices(indices, 1.0);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  
  static constexpr uint32_t PRIME_BASE = 257;
  static constexpr uint32_t PRIME_MOD = 1000000007;
  
  uint32_t _k;
  uint32_t _dim;
};

} // namespace thirdai::dataset
