#pragma once

#include <cstdint>

namespace thirdai::utils {

class HashUtils {
 public:
  /*
   * Cheap hash of two numbers - n1 and n2 - given seed and
   * bit range.
   */
  static constexpr uint32_t RandDoubleHash(int n1, int n2,
                                           uint32_t rand_double_hash_seed,
                                           uint32_t bit_range) {
    return (rand_double_hash_seed * (((n1 + 1) << 6) + n2) << 3) >>
           (32 - bit_range);
  }
};

}  // namespace thirdai::utils