#pragma once

#include <cereal/types/polymorphic.hpp>
#include "HashUtils.h"
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <stdexcept>

namespace thirdai::hashing {

class HashFunction {
 public:
  // TODO(any): Add comments

  explicit HashFunction(uint32_t num_tables, uint32_t range)
      : _num_tables(num_tables), _range(range) {}

  /**
   * Populates `num_tables` number of hashes for each element in the dataset
   * into the output array. The output array should be of size num_tables *
   * batch_size.
   *
   * The output array should be in vector major order. It should return all of
   * the hashes from the first vector, all of the hashes from the second, and
   * so on.
   */
  std::vector<uint32_t> hashBatchParallel(const BoltBatch& batch) const {
    std::vector<uint32_t> result(_num_tables * batch.getBatchSize());
    hashBatchParallel(batch, result.data());
    return result;
  }

  void hashBatchParallel(const BoltBatch& batch, uint32_t* output) const {
#pragma omp parallel for default(none) shared(batch, output)
    for (uint32_t v = 0; v < batch.getBatchSize(); v++) {
      if (batch[v].isDense()) {
        hashSingleDense(batch[v].activations, batch[v].len,
                        output + v * _num_tables);
      } else {
        hashSingleSparse(batch[v].active_neurons, batch[v].activations,
                         batch[v].len, output + v * _num_tables);
      }
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

  virtual std::unique_ptr<HashFunction> copyWithNewSeeds() const = 0;

  virtual ar::ConstArchivePtr toArchive() const {
    throw std::runtime_error("Cannot convert hash function to Archive.");
  }

  virtual std::string getName() const = 0;

  virtual ~HashFunction() = default;

 protected:
  uint32_t _num_tables, _range;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_tables, _range);
  }
};

using HashFunctionPtr = std::shared_ptr<HashFunction>;

}  // namespace thirdai::hashing