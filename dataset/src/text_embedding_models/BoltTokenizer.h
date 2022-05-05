#include <algorithm>
#pragma once

#include <limits>
#include <sstream>
#include <hashing/src/MurmurHash.h>
#include "TextEmbeddingModelInterface.h"

namespace thirdai::dataset {

struct BoltTokenizer: public TextEmbeddingModel {
  
  BoltTokenizer(uint32_t seed, uint32_t feature_dim): _seed(seed), _dim(feature_dim) {}

  void embedText(const std::string& text, BuilderVector& shared_feature_vector, uint32_t idx_offset) final {
    std::vector<uint32_t> indices;
    std::stringstream ss;
    ss << text;
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> tokens(begin, end);
    for (auto& s : tokens) {
      const char* cstr = s.c_str();
      uint32_t hash =
        thirdai::hashing::MurmurHash(cstr, s.length(), _seed) % _dim + idx_offset;
      indices.push_back(hash);
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
        shared_feature_vector.addSingleFeature(last_idx, last_idx_val);
        last_idx_val = 0.0;
      }
      last_idx = idx;
      last_idx_val += inc;
    }
  }
  
  uint32_t _seed;
  uint32_t _dim;
};

} // namespace thirdai::dataset