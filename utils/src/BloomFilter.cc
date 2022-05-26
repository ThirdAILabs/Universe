#include "BloomFilter.h"
#include <iostream>
#include <cmath>
#include <random>

namespace thirdai::utils {
template class BloomFilter<std::string>;

template <typename KEY_T>
BloomFilter<KEY_T>::BloomFilter(uint64_t capacity, float fp_rate, uint64_t input_dim=1):
  _fp_rate(fp_rate),
  _capacity(capacity),
  _count(0),
  _input_dim(input_dim){
  // R = log_(0.618)(fp) * N
  _R = (uint64_t) ceil(std::log(_fp_rate)/std::log(0.618) * _capacity);

  _K = (uint64_t) ceil(_R / _capacity * std::log(2));
  std::cout << "R: " << _R << "  K: " << _K << std::endl;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(1, 100000);
  for (uint64_t i = 0; i < _K; i++) {
    _seed_array.push_back(uni(rng));
  }
  //Bitarray
  _bit_array.resize(_R, 0);
  std::cout << _bit_array.at(0) << std::endl;
}

template <typename KEY_T>
void BloomFilter<KEY_T>::add(KEY_T key){
  if (_count > _capacity) {
    throw std::logic_error("Bloom Filter is at capacity");
  }
}

template <typename KEY_T>
std::vector<uint64_t> BloomFilter<KEY_T>::make_hashes(KEY_T key) {
  //TODO(henry): Add checking key data type
  std::string insert = static_cast<std::string>(key);
  // const char* cstr = insert.c_str();
  std::vector<uint64_t> hashes;
  return hashes;
}


template <typename KEY_T>
uint64_t BloomFilter<KEY_T>::size() {
  return _count;
}

template <typename KEY_T>
uint64_t BloomFilter<KEY_T>::capacity() {
  return _capacity;
}


}  // namespace thirdai::utils