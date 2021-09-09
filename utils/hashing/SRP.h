// #pragma once

#include "HashFunction.h"
#include <cstdint>
#include <vector>

namespace thirdai::utils {

class SparseRandomProjection : public HashFunction {
 private:
  uint32_t _K, _L, _num_hashes, _range, _dim;
  uint32_t _sample_size;
  short** _random_bits;
  uint32_t** _hash_indices;
  uint32_t _ratio = 3;

  void hashDenseVector(uint32_t index, float** values, uint32_t num_hashes,
                       uint32_t* output);

  void hashSparseVector(uint32_t index, uint32_t** indices, float** values,
                        const uint32_t* lengths, uint64_t num_hashes,
                        uint32_t* output);

 public:
  SparseRandomProjection(uint32_t input_dim, uint32_t _K, uint32_t _L,
                         uint32_t range_pow);

  void hashSparse(uint64_t num_vectors, uint32_t** indices, float** values,
                  uint32_t* lengths, uint64_t num_hashes, uint32_t* output);

  void hashDense(uint64_t num_vectors, uint64_t dim, float** values,
                 uint32_t num_hashes, uint32_t* output);

  void CompactHashes(uint32_t* hashes, uint32_t* final_hashes);

  ~SparseRandomProjection();
};

}  // namespace thirdai::utils
