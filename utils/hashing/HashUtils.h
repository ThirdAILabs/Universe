#pragma once

#include <cstdint>

namespace thirdai::utils {

class HashUtils {
 public:
  /*
   * Cheap hash of two numbers - n1 and n2 - given seed and
   * bit range.
   */
  static constexpr uint32_t randDoubleHash(int n1, int n2,
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
  static constexpr uint32_t fastDoubleHash(uint32_t num_one, uint32_t num_two,
                                           uint32_t bit_range) {
    const uint32_t prime = 92821;
    uint32_t result = prime * (prime * num_one + num_two);
    return result >> (32 - bit_range);
  }
};

}  // namespace thirdai::utils