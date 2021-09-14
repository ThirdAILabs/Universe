// #pragma once

#include "HashFunction.h"
#include <cstdint>
#include <vector>

namespace thirdai::utils {

class SparseRandomProjection : public HashFunction {
 private:
  const uint32_t _srps_per_table, _num_tables, _total_num_srps, _range, _dim,
      _sample_size;
  int16_t* _random_bits;
  uint32_t* _hash_indices;
  double _ratio = 0.3;

 public:
  SparseRandomProjection(uint32_t input_dim, uint32_t srps_per_table,
                         uint32_t num_tables, uint32_t range_pow);

  void hashSparse(uint64_t num_vectors, uint32_t** indices, float** values,
                  uint32_t* lengths, uint32_t* output) const override;

  void hashDense(uint64_t num_vectors, uint64_t dim, float** values,
                 uint32_t* output) const override;

  ~SparseRandomProjection();

  uint32_t numTables() const override { return _num_tables; }

  uint32_t range() const override { return _range; }
};

}  // namespace thirdai::utils
