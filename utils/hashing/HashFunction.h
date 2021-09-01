#pragma once

#include "../dataset/Dataset.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace thirdai::utils {

class HashFunction {
  /**
   * Populates num_hashes number of hashes for each element in the dataset into
   * the output array. The output array should be of size
   * num_hashes * batch_size, and vector i's hashes are stored in positions
   * i * num_hashes to i * (num_hashes) - 1.
   *
   */
  void hashBatch(Batch batch, uint64_t num_hashes, uint32_t* output) const {
    if (batch._type == BATCH_TYPE::SPARSE ||
        batch._type == BATCH_TYPE::SPARSE_LABELED) {
      hashSparse(batch._batch_size, batch._indices, batch._values, batch._lens,
                 num_hashes, output);
    } else {
      hashDense(batch._batch_size, batch._dim, batch._values, num_hashes,
                output);
    }
  }

  // TODO: Add comments
  virtual void hashSparse(uint64_t numVectors, uint32_t** indices,
                          float** values, uint32_t* lengths,
                          uint64_t num_hashes, uint32_t* output) const = 0;

  virtual void hashDense(uint64_t numVectors, uint64_t dim, float** values,
                         uint32_t numHashes, uint32_t* output) const = 0;
};

}  // namespace thirdai::utils