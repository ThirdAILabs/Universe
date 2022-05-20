#include "BloomFilter.h"
#include <iostream>
#include <cmath>
#include <random>

namespace thirdai::utils {

template <typename KEY_T>
BloomFilter<KEY_T>::BloomFilter(uint64_t capacity, uint64_t fp_rate) {
  _capacity = capacity;
  _fp_rate = fp_rate;
  // R = log_(0.618)(fp) * N
  _R = (uint64_t) ceil(log(fp_rate)/log(0.618) * capacity);
  _K = (uint64_t) ceil(_R / capacity * log(2));
  count = 0;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(1, 100000);
  for (uint64_t i = 0; i < _K; i++) {
    // _seed_array.push_back(i);
    _seed_array.push_back(uni(rng));
  }
  //TODO: Bitarray
}

}  // namespace thirdai::utils