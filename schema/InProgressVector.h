#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace thirdai::schema {

struct InProgressSparseVector {

  InProgressSparseVector() {}

  void addSingleFeature(uint32_t index, float value) {
    _indices.push_back(index);
    _values.push_back(value);
  }

  void addSparseFeatures(std::vector<uint32_t>& indices, std::vector<float>& values) {
    assert(indices.size() == values.size());
    _indices.insert(_indices.end(), indices.begin(), indices.end());
    _values.insert(_values.end(), values.begin(), values.end());
  }

  void addDenseFeatures(uint32_t start_idx, std::vector<float>& values) {
    _indices.reserve(_indices.size() + values.size());
    for (uint32_t idx = start_idx; idx < start_idx + values.size(); idx++) {
      _indices.push_back(idx);
    }
    _values.insert(_values.end(), values.begin(), values.end());
  }

  void incrementAtIndices(std::vector<uint32_t>& indices, uint32_t inc) {
    std::sort(indices.begin(), indices.end());
    uint32_t last_idx = std::numeric_limits<uint32_t>::max();
    for (const auto& idx : indices) {
      if (idx != last_idx) {
        addSingleFeature(idx, inc);
      } else {
        _values.back() += inc;
      }
    }
  }

  bolt::BoltVector toBoltVector() {
    return bolt::BoltVector::makeSparseVector(_indices, _values);
  }

 private:
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
  
};

} // namespace thirdai::schema