#pragma once

#include "BoltTokenBatch.h"
#include <bolt/src/layers/BoltVector.h>
#include <vector>

namespace thirdai::dataset {

using bolt::BoltVector;

class MaskedSentenceBatch {
 public:
  MaskedSentenceBatch(std::vector<BoltVector>&& vectors,
                      std::vector<std::vector<uint32_t>>&& positions)
      : _input_vectors(std::move(vectors)),
        _masked_positions(std::move(positions)) {}

  uint32_t getBatchSize() const { return _input_vectors.getBatchSize(); }

  const BoltVector& operator[](size_t i) const {
    assert(i < _input_vectors.getBatchSize());
    return _input_vectors[i];
  }

  BoltVector& operator[](size_t i) {
    assert(i < _input_vectors.getBatchSize());
    return _input_vectors[i];
  }

  // The BoltTokenBatch uses a vector of uint32_t for generalizability to
  // multiple tokens. Here we know that there is only one masked index per
  // vector so we can create a convenience method to simplify accessing it.
  uint32_t maskedIndex(size_t i) {
    assert(i < _input_vectors.getBatchSize());
    return _masked_positions[i].at(0);
  }

  bolt::BoltBatch* getVectors() { return &_input_vectors; }

 private:
  bolt::BoltBatch _input_vectors;
  BoltTokenBatch _masked_positions;
};

}  // namespace thirdai::dataset