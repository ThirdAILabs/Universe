#pragma once

#include "../Vectors.h"
#include <cassert>
#include <fstream>

namespace thirdai::utils {

template <typename Id_t>
class SvmBatch {
 public:
  explicit SvmBatch(std::ifstream& file, uint32_t target_batch_size,
                    Id_t start_id);

  const SparseVector& operator[](uint32_t i) const {
    assert(i < _batch_size);
    return _vectors[i];
  }

  const uint32_t* labels(uint32_t i) const {
    assert(i < _batch_size);
    return _labels[i];
  }

  // TODO(Nicholas, Josh, Geordie): should this be a template or
  // uint32_t/uint64_t
  Id_t id(uint32_t i) const {
    assert(i < _batch_size);
    return _start_id + i;
  }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  SparseVector* _vectors;
  uint32_t _batch_size;
  uint32_t** _labels;
  Id_t _start_id;
};

}  // namespace thirdai::utils
