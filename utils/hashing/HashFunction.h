#pragma once

#include "../dataset/Dataset.h"
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
  void hashBatchParallel(const Batch& batch, uint32_t* output) const {
    if (batch._batch_type == BATCH_TYPE::SPARSE) {
      hashSparseParallel(batch._batch_size, batch._indices, batch._values,
                         batch._lens, output);
    } else {
      hashDenseParallel(batch._batch_size, batch._values, batch._dim, output);
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

  void hashBatchSerial(const Batch& batch, uint32_t* output) const {
    if (batch._batch_type == BATCH_TYPE::SPARSE) {
      hashSparseSerial(batch._batch_size, batch._indices, batch._values,
                       batch._lens, output);
    } else {
      hashDenseParallel(batch._batch_size, batch._values, batch._dim, output);
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

  /**
   * Performas a default hash compation by assigning hashes_per_output_value
   * hashes to each of the length_output output hashes. If
   * length_output * hashes_per_output_value >= len(hashes), this will segfault,
   * so don't do that.
   */
  static void defaultCompactHashesMethod(const uint32_t* hashes,
                                         uint32_t* output_hashes,
                                         uint32_t length_output,
                                         uint32_t hashes_per_output_value) {
    for (uint32_t i = 0; i < length_output; i++) {
      uint32_t index = 0;
      for (uint32_t j = 0; j < hashes_per_output_value; j++) {
        uint32_t h = hashes[i * hashes_per_output_value + j];
        index += h << (hashes_per_output_value - 1 - j);
      }
      output_hashes[i] = index;
    }
  }

  /**
   * Does an in place densification of hashes, as described in the DOPH paper.
   * Currently unset hashes should be represented by UINT32_MAX. For a given
   * unset hash, if we don't find a set hash within max_path_length number of
   * jumps, we set it to unset_hash_value. First performs a densification of
   * the largest 2^x number of hashes less than the number of hashes, then
   * densifies the rest.
   */
  static void densifyHashes(uint32_t* hashes, uint32_t num_hashes,
                            uint32_t max_path_length = 100,
                            uint32_t unset_hash_value = 0) {
    if (num_hashes == 0) {
      return;
    }

    // TODO(josh): Make this a util log method. __builtin_clz returns
    // the number of zeros before the first set bit, so the log is 32 - 1 -
    // this number.
    const uint32_t log_2_floor = 31 - __builtin_clz(num_hashes);
    const uint32_t densify_hashes_block_length = 1 << log_2_floor;
    const uint32_t num_hashes_in_overlap =
        2 * densify_hashes_block_length - num_hashes;

    densifyHashesPowerOf2(hashes, log_2_floor, 0, max_path_length,
                          unset_hash_value);
    densifyHashesPowerOf2(
        hashes + densify_hashes_block_length - num_hashes_in_overlap,
        log_2_floor, num_hashes_in_overlap, max_path_length, unset_hash_value);
  }

 private:
  /**
   * Does an in place densification of a power of 2 number of hashes. Starts
   * at an optional offset in case we already know that some of the hashes are
   * filled.
   */
  static void densifyHashesPowerOf2(uint32_t* hashes, uint32_t log_num_hashes,
                                    uint32_t starting_offset,
                                    uint32_t max_path_length,
                                    uint32_t unset_hash_value) {
    uint32_t num_hashes = 1 << log_num_hashes;

    // Do the first largest_power_of_2_hashes hashes
    for (uint32_t i = starting_offset; i < num_hashes; i++) {
      uint32_t final_hash = hashes[i];
      uint32_t count = 0;
      while (final_hash == UINT32_MAX) {
        count++;
        uint32_t index = HashUtils::fastDoubleHash(i, count, log_num_hashes);

        final_hash = hashes[index];
        if (count > max_path_length) {  // Densification failure.
          final_hash = unset_hash_value;
          break;
        }
      }
      hashes[i] = final_hash;
    }
  }
};

}  // namespace thirdai::utils