#pragma once

#include "../dataset/Dataset.h"

namespace thirdai::utils {

class HashFunction {
 public:
  /**
   * Populates num_hashes number of hashes for each element in the dataset into
   * the output array. The output array should be of size
   * num_hashes * batch_size, and vector i's hashes are stored in positions
   * i * num_hashes to i * (num_hashes) - 1.
   *
   */
  void hashBatch(const Batch& batch, uint64_t num_hashes, uint32_t* output) const {
    if (batch._type == BATCH_TYPE::SPARSE ||
        batch._type == BATCH_TYPE::SPARSE_LABELED) {
      hashSparse(batch._batch_size, batch._indices, batch._values, batch._lens,
                 num_hashes, output);
    } else {
      hashDense(batch._batch_size, batch._dim, batch._values, num_hashes,
                output);
    }
  }

  void hashSingleSparse(uint32_t* indices, float* values, uint32_t length,
                        uint64_t num_hashes, uint32_t* output) {
    uint32_t lengths[1] = {length};
    hashSparse(1, &indices, &values, lengths, num_hashes, output);
  }

  void hashSingleDense(float* values, uint32_t dim, uint64_t num_hashes,
                       uint32_t* output) {
    hashDense(1, dim, &values, num_hashes, output);
  }

  // TODO: Add comments
  virtual void hashSparse(uint64_t numVectors, uint32_t** indices,
                          float** values, uint32_t* lengths,
                          uint64_t num_hashes, uint32_t* output) const = 0;

  virtual void hashDense(uint64_t numVectors, uint64_t dim, float** values,
                         uint32_t numHashes, uint32_t* output) const = 0;
};

}  // namespace thirdai::utils