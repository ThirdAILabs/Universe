#pragma once

#include "../dataset/batch_types/CsvBatch.h"
#include "../dataset/batch_types/SvmBatch.h"
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
  template <typename Id_t>
  void hashBatchParallel(const utils::SvmBatch<Id_t>& batch,
                         uint32_t* output) const {
#pragma omp parallel for default(none) shared(batch, output)
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      hashSingleSparse(batch[v].indices, batch[v].values, batch[v].len,
                       output + v * _num_tables);
    }
  }

  template <typename Id_t>
  void hashBatchParallel(const utils::CsvBatch<Id_t>& batch,
                         uint32_t* output) const {
#pragma omp parallel for default(none) shared(batch, output)
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      hashSingleDense(batch[v].values, batch[v].dim, output + v * _num_tables);
    }
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

  template <typename Batch_t>
  void hashBatchSerial(const Batch_t& batch, uint32_t* output) const {
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      if (std::is_same<Batch_t, SvmBatch<uint32_t>>::value ||
          std::is_same<Batch_t, SvmBatch<uint64_t>>::value) {
        hashSingleSparse(batch[v].indices, batch[v].values, batch[v].len,
                         output + v * _num_tables);
      } else if (std::is_same<Batch_t, CsvBatch<uint32_t>>::value ||
                 std::is_same<Batch_t, CsvBatch<uint64_t>>::value) {
        hashSingleDense(batch[v].values, batch[v].dim,
                        output + v * _num_tables);
      }
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