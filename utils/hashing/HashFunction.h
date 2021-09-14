#pragma once

#include "../dataset/Dataset.h"

namespace thirdai::utils {

class HashFunction {
 public:
  /**
   * Populates num_hashes number of hashes for each element in the dataset into
   * the output array. The output array should be of size
   * num_hashes * batch_size.
   *
   * Should return all of the first hashes, followed by all of the second
   * hashes, etc, instead of all of the first vectors’ hashes, all of the second
   * vectors’ hashes, etc.
   *
   */
  void hashBatch(const Batch& batch, uint32_t* output) const {
    if (batch._batch_type == BATCH_TYPE::SPARSE) {
      hashSparse(batch._batch_size, batch._indices, batch._values, batch._lens,
                 output);
    } else {
      hashDense(batch._batch_size, batch._dim, batch._values, output);
    }
  }

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const {
    uint32_t lengths[1] = {length};
    hashSparse(1, &indices, &values, lengths, output);
  }

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const {
    hashDense(1, dim, &values, output);
  }

  // TODO(any): Add comments
  virtual void hashSparse(uint64_t num_vectors, const uint32_t* const* indices,
                          const float* const* values, const uint32_t* lengths,
                          uint32_t* output) const = 0;

  virtual void hashDense(uint64_t num_vectors, uint64_t dim,
                         const float* const* values,
                         uint32_t* output) const = 0;

  virtual uint32_t numTables() const = 0;

  virtual uint32_t range() const = 0;
};

}  // namespace thirdai::utils