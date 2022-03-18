#pragma once

#include "BoltVector.h"

namespace thirdai::dataset {

using bolt::BoltVector;

// TODO(Geordie): Bolt input batch must also support lack of labels.

class BoltInputBatch {
 public:
  BoltInputBatch(std::vector<BoltVector>&& vectors,
                 std::vector<BoltVector>&& labels)
      : _vectors(std::move(vectors)), _labels(std::move(labels)) {}

  uint32_t getBatchSize() const { return _vectors.size(); }

  const BoltVector& operator[](size_t i) const {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  BoltVector& operator[](size_t i) {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  const BoltVector& labels(size_t i) const {
    assert(i < _labels.size());
    return _labels[i];
  }

 private:
  std::vector<BoltVector> _vectors;
  std::vector<BoltVector> _labels;
};

}  // namespace thirdai::dataset