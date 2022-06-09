#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <vector>

namespace thirdai::dataset {

using bolt::BoltVector;

class MaskedSentenceBatch {
 public:
  MaskedSentenceBatch(std::vector<BoltVector>&& vectors,
                      std::vector<uint32_t>&& positions)
      : _input_vectors(vectors), _masked_positions(positions) {}

  uint32_t getBatchSize() const { return _input_vectors.size(); }

  const BoltVector& operator[](size_t i) const {
    assert(i < _input_vectors.size());
    return _input_vectors[i];
  }

  BoltVector& operator[](size_t i) {
    assert(i < _input_vectors.size());
    return _input_vectors[i];
  }

  uint32_t maskedIndex(size_t i) const { return _masked_positions[i]; }

 private:
  std::vector<BoltVector> _input_vectors;
  std::vector<uint32_t> _masked_positions;
};

}  // namespace thirdai::dataset