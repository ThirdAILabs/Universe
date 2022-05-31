#include <hashing/src/MurmurHash.h>
#include <iostream>
#include <vector>
#include <cstring>
 

namespace thirdai::utils {

template <typename KEY_T>
class BloomFilter {
 private:
  float _fp_rate;
  uint64_t _capacity, _R, _K, _count, _input_dim;
  std::vector<uint32_t> _seed_array;
  // implementation defined, vector<bool> can store bits instead of <bool> type
  // elements
  std::vector<bool> _bit_array;

 public:
  /*
      this BloomFilter must be able to store at least *capacity* elements
      while maintaining no more than *fp_rate* chance of false
      positives
  */
  BloomFilter(uint64_t capacity, float fp_rate, uint64_t input_dim = 1);

  BloomFilter(const BloomFilter& other) = delete;

  BloomFilter& operator=(const BloomFilter& other) = delete;

  std::vector<uint64_t> make_hashes(const KEY_T& key);

  /*
      This bloom filter currently supports storage of type std::string.
      TODO(Henry): Add storage of type std::vector<int>
  */
  void add(const KEY_T& key, bool skip_check = false);

  // bool contains(const KEY_T& query);

  // void clear();

  // TODO: Figure out the best way to do this
  // BloomFilter<KEY_T> intersection(BloomFilter other);

  // BloomFilter<KEY_T> union(BloomFilter other);

  uint64_t size();

  uint64_t capacity();

  ~BloomFilter() = default;
};

}  // namespace thirdai::utils