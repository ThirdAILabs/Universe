#pragma once

#include <algorithm>
#include <cctype>
#include <ctype.h>
#include <limits>
#include <sstream>
#include <hashing/src/MurmurHash.h>
#include "TextEmbeddingModelInterface.h"

namespace thirdai::dataset {

struct BoltTokenizer: public TextEmbeddingModel {
  
  BoltTokenizer(uint32_t seed=42, uint32_t feature_dim=100000): _seed(seed), _dim(feature_dim) {}

  void embedText(const std::string& text, BuilderVector& shared_feature_vector, uint32_t idx_offset) final {
    std::string lower_case_text = text;
    for (auto& c : lower_case_text) {
      c = std::tolower(c);
    }
    std::vector<uint32_t> indices;

    const auto *start_ptr = lower_case_text.c_str();
    bool last_ptr_was_space = false;
    for (const auto& c : lower_case_text) {
      if (isspace(c) && !last_ptr_was_space) {
        last_ptr_was_space = true;

        uint32_t hash =
          hashing::MurmurHash(start_ptr, &c - start_ptr, _seed) % _dim + idx_offset;
        indices.push_back(hash);
      }
      if (last_ptr_was_space && !isspace(c)) {
        last_ptr_was_space = false;
        start_ptr = &c;
      }
    }

    shared_feature_vector.incrementAtIndices(indices, 1.0);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  
  uint32_t _seed;
  uint32_t _dim;
};

} // namespace thirdai::dataset