#include "BloomFilter.h"
#include <cmath>
#include <iostream>
#include <random>

namespace thirdai::utils {
template class BloomFilter<std::string>;

template <typename KEY_T>
BloomFilter<KEY_T>::BloomFilter(uint64_t capacity, float fp_rate,
                                uint64_t input_dim)
    : _fp_rate(fp_rate), _capacity(capacity), _count(0), _input_dim(input_dim) {
  // R = log_(0.618)(fp) * N
  _R = (uint64_t)ceil(std::log(_fp_rate) / std::log(0.618) * _capacity);

  _K = (uint64_t)ceil(_R / _capacity * std::log(2));
  std::cout << "R: " << _R << "  K: " << _K << std::endl;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(1, 100000);
  for (uint64_t i = 0; i < _K; i++) {
    _seed_array.push_back(uni(rng));
  }
  // Bitarray
  _bit_array.resize(_R, 0);
  std::cout << _bit_array.at(0) << std::endl;
}

template <typename KEY_T>
void BloomFilter<KEY_T>::add(const KEY_T& key, bool skip_check) {
  if (_count > _capacity) {
    throw std::logic_error("Bloom Filter is at capacity");
  }
  std::vector<uint64_t> hashes = make_hashes(key);
  bool is_in = true;
  for (uint64_t i = 0; i < _K; i++) {
    if (!skip_check && is_in && _bit_array.at(hashes.at(i)) == 0) {
      is_in = false;
    }
    _bit_array.at(hashes.at(i)) = 1;
  }

  if (skip_check || !is_in) {
    _count++;
  }
}

template <typename KEY_T>
std::vector<uint64_t> BloomFilter<KEY_T>::make_hashes(const KEY_T& key) {
  // TODO(henry): Add checking key data type
  const char* cstr = static_cast<std::string>(key).c_str();
  std::vector<uint64_t> hashes;
  for (uint64_t i = 0; i < _K; i++) {
    hashes.push_back(thirdai::hashing::MurmurHash(cstr, strlen(cstr),
                                                   _seed_array.at(i)) % _R);
  }
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