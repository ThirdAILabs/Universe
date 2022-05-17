#pragma once

#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace thirdai::hashing {

class HashUtils {
 public:
  /*
   * Cheap hash of two numbers - n1 and n2 - given seed and
   * bit range.
   */
  static constexpr uint32_t randDoubleHash(uint32_t n1, uint32_t n2,
                                           uint32_t rand_double_hash_seed,
                                           uint32_t bit_range) {
    return (rand_double_hash_seed * (((n1 + 1) << 6) + n2) << 3) >>
           (32 - bit_range);
  }

  /*
   * A very cheap and fast hash of two numbers into a hash of a certain bit
   * range. Inspiration from
   * https://stackoverflow.com/questions/1835976/what-is-a-sensible-prime-for-hashcode-calculation/2816747#2816747
   */
  static constexpr uint32_t fastDoubleHash(uint32_t n1, uint32_t n2,
                                           uint32_t bit_range) {
    const uint32_t prime = 92821;
    uint32_t result = prime * (prime * n1 + n2);
    return result >> (32 - bit_range);
  }

  /**
   * Performs a default hash compaction by assigning hashes_per_output_value
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
   * Performas a default hash compaction by compacting hashes into each of the
   * output bins. If length_output *  hashes_per_output_value >= len(hashes),
   * this will segfault, so don't do that. The range of this is UINT32_MAX.
   */
  static void defaultCompactHashes(const uint32_t* hashes,
                                   uint32_t* output_hashes,
                                   uint32_t length_output,
                                   uint32_t hashes_per_output_value) {
    for (uint32_t i = 0; i < length_output; i++) {
      uint32_t current_hash = hashes[i * hashes_per_output_value];
      for (uint32_t j = 1; j < hashes_per_output_value; j++) {
        // See
        // https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
        // for explanation of the magic number (this is the boost
        // implementation of combining hash values).
        current_hash ^= hashes[i * hashes_per_output_value + j] + 0x9e3779b9 +
                        (current_hash << 6) + (current_hash >> 2);
      }
      output_hashes[i] = current_hash;
    }
  }

  /**
   * Performas a default hash compaction by assigning hashes_per_output_value
   * hash bits to each of the length_output output hashes. Assumes that
   * the input hashes are all either 0 or 1. If length_output *
   * hashes_per_output_value >= len(hashes), this will segfault,
   * so don't do that.
   */
  static void compactHashBits(const uint32_t* hashes, uint32_t* output_hashes,
                              uint32_t length_output,
                              uint32_t hashes_per_output_value) {
    for (uint32_t i = 0; i < length_output; i++) {
      uint32_t current_hash = 0;
      for (uint32_t j = 0; j < hashes_per_output_value; j++) {
        uint32_t h = hashes[i * hashes_per_output_value + j];
        current_hash += h << (hashes_per_output_value - 1 - j);
      }
      output_hashes[i] = current_hash;
    }
  }

  static uint32_t log_2_floor(uint32_t input) {
// If none of these this won't return anything so it won't compile.
#ifdef __GNUC__
    return 31 - __builtin_clz(input);
#elif _MSC_VER
    unsigned long index_first_set_bit;
    // Returns the index of the first set bit
    _BitScanReverse(&index_first_set_bit, input);
    return index_first_set_bit;
#endif
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
    const uint32_t log_2_floored = log_2_floor(num_hashes);
    const uint32_t densify_hashes_block_length = 1 << log_2_floored;
    const uint32_t num_hashes_in_overlap =
        2 * densify_hashes_block_length - num_hashes;

    densifyHashesPowerOf2(hashes, log_2_floored, 0, max_path_length,
                          unset_hash_value);
    densifyHashesPowerOf2(
        hashes + densify_hashes_block_length - num_hashes_in_overlap,
        log_2_floored, num_hashes_in_overlap, max_path_length,
        unset_hash_value);
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
        uint32_t index = fastDoubleHash(i, count, log_num_hashes);

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

}  // namespace thirdai::hashing