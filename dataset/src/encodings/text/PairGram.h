#pragma once

#include <algorithm>
#include <cctype>
#include <ctype.h>
#include <limits>
#include <sstream>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include "TextEncodingInterface.h"

namespace thirdai::dataset {

struct PairGram: public TextEncoding {
  
  explicit PairGram(uint32_t dim=100000): _dim(dim) {}

  void embedText(const std::string& text, BuilderVector& shared_feature_vector, uint32_t idx_offset) final {
    std::string lower_case_text = text;
    for (auto& c : lower_case_text) {
      c = std::tolower(c);
    }
    std::vector<uint32_t> indices;
    std::vector<uint32_t> hashes;

    const auto *start_ptr = lower_case_text.c_str();
    bool last_ptr_was_space = false;
    for (const auto& c : lower_case_text) {
      if (isspace(c) && !last_ptr_was_space) {
        last_ptr_was_space = true;

        uint32_t hash =
          hashing::MurmurHash(start_ptr, &c - start_ptr, /* seed = */ 341) % _dim + idx_offset;
        hashes.push_back(hash);

        for (const auto& prev_word_hash : hashes) {
            indices.push_back(hashing::HashUtils::combineHashes(prev_word_hash, hash));
        }    
      }
      if (last_ptr_was_space && !isspace(c)) {
        last_ptr_was_space = false;
        start_ptr = &c;
      }
    }

    shared_feature_vector.incrementAtIndices(hashes, 1.0);
  }

  uint32_t featureDim() final { return _dim; }

  bool isDense() final { return false; }

 private:
  
  uint32_t _dim;
};

} // namespace thirdai::dataset