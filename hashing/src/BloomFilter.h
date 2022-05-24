#pragma once

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace thirdai::hashing {

class BloomFilter {
 public:
  BloomFilter(uint64_t num_hashes, uint64_t requested_total_num_bits,
              uint64_t input_dim);

  void add(const std::vector<uint64_t>& input);

  bool is_present(const std::vector<uint64_t>& input);
};

}  // namespace thirdai::hashing