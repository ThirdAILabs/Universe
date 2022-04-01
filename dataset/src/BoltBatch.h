#pragma once

#include "BoltVector.h"
#include <string>
#include <sstream>

namespace thirdai::dataset {

class BoltBatch {
 public:
  BoltBatch(std::vector<BoltVector>&& vectors)
      : _vectors(std::move(vectors)) {}

  uint32_t getBatchSize() const { return _vectors.size(); }

  const BoltVector& operator[](size_t i) const {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  BoltVector& operator[](size_t i) {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "========================================================================\n";
    ss << "Batch | size = " << _vectors.size() << "\n\n";
    for (size_t i = 0; i < _vectors.size(); i++) {
      ss << "Vector " << i << ": " << _vectors.at(i).toString() << "\n\n";
    }
    ss << "========================================================================";
    return ss.str();
  }

 private:
  std::vector<BoltVector> _vectors;
};

}  // namespace thirdai::dataset