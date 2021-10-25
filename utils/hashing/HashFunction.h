#pragma once

#include "../dataset/batch_types/DenseBatch.h"
#include "../dataset/batch_types/SparseBatch.h"
#include "HashUtils.h"

namespace thirdai::utils {

class HashFunction {
 public:
  // TODO(any): Add comments

  explicit HashFunction(uint32_t num_tables, uint32_t range)
      : _num_tables(num_tables), _range(range) {}

  /**
   * Populates num_hashes number of hashes for each element in the dataset into
   * the output array. The output array should be of size
   * num_hashes * batch_size.
   *
   * The output array should be in vector major order. It should return all of
   * the hashes from the first vector, all of the hashes from the second, and
   * so on.
   */
  void hashBatchParallel(const utils::SparseBatch& batch,
                         uint32_t* output) const {
#pragma omp parallel for default(none) shared(batch, output)
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      hashSingleSparse(batch[v]._indices, batch[v]._values, batch[v]._len,
                       output + v * _num_tables);
    }
  }

  std::vector<uint32_t> hashBatchParallel(
      const utils::SparseBatch& batch) const {
    std::vector<uint32_t> result(_num_tables * batch.getBatchSize());
    hashBatchParallel(batch, result.data());
    return result;
  }

  void hashBatchParallel(const utils::DenseBatch& batch,
                         uint32_t* output) const {
#pragma omp parallel for default(none) shared(batch, output)
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      hashSingleDense(batch[v]._values, batch[v]._dim,
                      output + v * _num_tables);
    }
  }

  std::vector<uint32_t> hashBatchParallel(
      const utils::DenseBatch& batch) const {
    std::vector<uint32_t> result(_num_tables * batch.getBatchSize());
    hashBatchParallel(batch, result.data());
    return result;
  }

  void hashSparseParallel(uint64_t num_vectors, const uint32_t* const* indices,
                          const float* const* values, const uint32_t* lengths,
                          uint32_t* output) const {
#pragma omp parallel for default(none) \
    shared(num_vectors, indices, values, lengths, output)
    for (uint32_t v = 0; v < num_vectors; v++) {
      hashSingleSparse(indices[v], values[v], lengths[v],
                       output + v * _num_tables);
    }
  }

  void hashDenseParallel(uint64_t num_vectors, const float* const* values,
                         uint32_t dim, uint32_t* output) const {
#pragma omp parallel for default(none) shared(num_vectors, values, output, dim)
    for (uint32_t v = 0; v < num_vectors; v++) {
      hashSingleDense(values[v], dim, output + v * _num_tables);
    }
  }

  void hashBatchSerial(const utils::SparseBatch& batch,
                       uint32_t* output) const {
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      hashSingleSparse(batch[v]._indices, batch[v]._values, batch[v]._len,
                       output + v * _num_tables);
    }
  }

  void hashBatchSerial(const utils::DenseBatch& batch, uint32_t* output) const {
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      hashSingleDense(batch[v]._values, batch[v]._dim,
                      output + v * _num_tables);
    }
  }

  void hashSparseSerial(uint64_t num_vectors, const uint32_t* const* indices,
                        const float* const* values, const uint32_t* lengths,
                        uint32_t* output) const {
    for (uint32_t v = 0; v < num_vectors; v++) {
      hashSingleSparse(indices[v], values[v], lengths[v],
                       output + v * _num_tables);
    }
  }

  void hashDenseSerial(uint64_t num_vectors, const float* const* values,
                       uint32_t dim, uint32_t* output) const {
    for (uint32_t v = 0; v < num_vectors; v++) {
      hashSingleDense(values[v], dim, output + v * _num_tables);
    }
  }

  virtual void hashSingleSparse(const uint32_t* indices, const float* values,
                                uint32_t length, uint32_t* output) const = 0;

  virtual void hashSingleDense(const float* values, uint32_t dim,
                               uint32_t* output) const = 0;

  inline uint32_t numTables() const { return _num_tables; }

  inline uint32_t range() const { return _range; }

 protected:
  const uint32_t _num_tables, _range;
};

}  // namespace thirdai::utils