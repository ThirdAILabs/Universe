#pragma once

#include "../dataset/Dataset.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace thirdAIUtils {

class HashFunction {
  /**
   * Populates num_hashes number of hashes for each element in the dataset into
   * the output array. The output array should be of size
   * num_hashes * batch_size, and vector i's hashes are stored in positions
   * i * num_hashes to i * (num_hashes) - 1.
   *
   */
  void getBatchHashes(Batch batch, uint64_t num_hashes, uint32_t* output) {
    if (batch._type == FORMAT_TYPE::SPARSE ||
        batch._type == FORMAT_TYPE::SPARSE_LABELED) {
      getSparseHashes(batch._batch_size, batch._indices, batch._values,
                      batch._lens, num_hashes, output);
    } else {
      getDenseHashes(batch._batch_size, batch._dim, batch._values, num_hashes,
                     output);
    }
  }

  // TODO: Add comments
  virtual void getSparseHashes(uint64_t numVectors, uint32_t** indices,
                               float** values, uint32_t* lengths,
                               uint64_t num_hashes, uint32_t* output) = 0;

  virtual void getDenseHashes(uint64_t numVectors, uint64_t dim, float** values,
                              uint32_t numHashes, uint32_t* output) = 0;
};

}  // namespace thirdAIUtils