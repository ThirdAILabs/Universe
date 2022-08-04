#include "CompressedVector.h"

namespace thirdai::bolt {

namespace {

// Convenience function to hash into a uint32_t using
// MurmurHash using saved seed value.
inline uint32_t hashFunction(uint64_t value, uint32_t seed) {
  char* addr = reinterpret_cast<char*>(&value);
  uint32_t hash_value =
      thirdai::hashing::MurmurHash(addr, sizeof(uint64_t), seed);
  return hash_value;
}

}  // namespace

// Create a new BiasedSketch.
template <class ELEMENT_TYPE>
BiasedSketch<ELEMENT_TYPE>::BiasedSketch(
    uint64_t physical_size, ELEMENT_TYPE default_value /* = 0*/,
    uint64_t block_size /* = kDefaultBlockSize*/,
    uint32_t seed /*= kDefaultSeed*/)
    : _physical_vector(physical_size + block_size, default_value),
      _block_size(block_size),
      _seed(seed),
      _truncated_size(physical_size) {}

// Create a new BiasedSketch from a pre-existing vector.
template <class ELEMENT_TYPE>
BiasedSketch<ELEMENT_TYPE>::BiasedSketch(
    const std::vector<ELEMENT_TYPE>& input, uint64_t physical_size,
    uint64_t block_size /* = kDefaultBlockSize*/,
    uint32_t seed /*= kDefaultSeed*/)
    : BiasedSketch(physical_size, /*default_value=*/
                   0, block_size, seed) {
  // Do we have BOLT_ASSERT yet?
  assert(physical_size <= input.size());
  assert(physical_size > block_size);

  for (uint64_t i = 0; i < input.size(); i += _block_size) {
    // Find the location the first element of the block hashes into.
    // Hashing is truncated by truncated_size to avoid out of bounds access in
    // the nested loop below.

    uint64_t block_begin = hashFunction(i, _seed) % _truncated_size;

    // Having found the hash, we store all elements in the block within the
    // respective offset.
    for (uint64_t j = i; j < i + _block_size; j++) {
      uint64_t offset = j - i;
      uint64_t index = block_begin + offset;

      _physical_vector[index] += input[j];
    }
  }
}

// non-const accessor.
template <class ELEMENT_TYPE>
ELEMENT_TYPE BiasedSketch<ELEMENT_TYPE>::get(uint64_t i) const {
  uint64_t idx = findIndexInPhysicalVector(i);
  ELEMENT_TYPE value = _physical_vector[idx];

  assert(not std::isnan(value));

  return value;
}

// Set a value at an index.
template <class ELEMENT_TYPE>
void BiasedSketch<ELEMENT_TYPE>::set(uint64_t i, ELEMENT_TYPE value) {
  uint64_t idx = findIndexInPhysicalVector(i);
  ELEMENT_TYPE& current_value = _physical_vector[idx];
  // current_value += value;
  current_value = value;
}

template <class ELEMENT_TYPE>
void BiasedSketch<ELEMENT_TYPE>::assign(uint64_t size, ELEMENT_TYPE value) {
  (void)size;
  std::fill(_physical_vector.data(),
            _physical_vector.data() + _physical_vector.size(), value);
}

template <class ELEMENT_TYPE>
void BiasedSketch<ELEMENT_TYPE>::clear() {
  _physical_vector.clear();
}

template <class ELEMENT_TYPE>
uint64_t BiasedSketch<ELEMENT_TYPE>::findIndexInPhysicalVector(
    uint64_t i) const {
  // The following involves the mod operation and is slow.
  // We will have to do bit arithmetic somewhere.
  // TODO(jerin): Come back here and make more efficient.
  uint64_t offset = i % _block_size;
  uint64_t i_begin = i - offset;

  uint64_t block_begin = hashFunction(i_begin, _seed) % _truncated_size;
  uint64_t index = block_begin + offset;
  return index;
}

template <class ELEMENT_TYPE>
uint64_t UnbiasedSketch<ELEMENT_TYPE>::findIndexInPhysicalVector(
    uint64_t i) const {
  // The following involves the mod operation and is slow.
  // We will have to do bit arithmetic somewhere.
  // TODO(jerin): Come back here and make more efficient.
  uint64_t offset = i % _block_size;
  uint64_t i_begin = i - offset;

  uint64_t block_begin = hashFunction(i_begin, _seed) % _truncated_size;
  uint64_t index = block_begin + offset;
  return index;
}

// Create a new UnbiasedSketch from a pre-existing vector.
template <class ELEMENT_TYPE>
UnbiasedSketch<ELEMENT_TYPE>::UnbiasedSketch(
    const std::vector<ELEMENT_TYPE>& input, uint64_t physical_size,
    uint64_t block_size /* = kDefaultBlockSize*/,
    uint32_t seed /* = kDefaultSeed*/)
    : UnbiasedSketch(physical_size, /*default_value=*/0, block_size, seed) {
  // Do we have BOLT_ASSERT yet?
  assert(physical_size <= input.size());
  assert(physical_size > block_size);

  for (uint64_t i = 0; i < input.size(); i += _block_size) {
    // Find the location the first element of the block hashes into.
    // Hashing is truncated by truncated_size to avoid out of bounds access in
    // the nested loop below.

    uint64_t block_begin = hashFunction(i, _seed) % _truncated_size;

    // Having found the hash, we store all elements in the block within the
    // respective offset.
    for (uint64_t j = i; j < i + _block_size; j++) {
      uint64_t offset = j - i;
      uint64_t index = block_begin + offset;

      bool sign_bit = hashFunction(j, _seed) % 2;

      // Add the input value multiplied by sign bit to the index at
      // _physical_vector.
      if (sign_bit) {
        _physical_vector[index] += input[j];
      } else {
        _physical_vector[index] -= input[j];
      }

      // TODO(jerin): What happens if overflow? We are using sum to store
      // multiple elements, which could overflow the element's type.
    }
  }

  // for (size_t i = 0; i < input.size(); i++) {
  //   if (input[i] != get(i)) {
  //     std::cout << i << ": " << input[i] << ", " << get(i) << "\n";
  //   }
  // }
}

template <class ELEMENT_TYPE>
ELEMENT_TYPE UnbiasedSketch<ELEMENT_TYPE>::get(uint64_t i) const {
  uint64_t idx = findIndexInPhysicalVector(i);
  ELEMENT_TYPE value = _physical_vector[idx];

  uint64_t sign_bit = hashFunction(i, _seed) % 2;
  value = sign_bit ? value : -1 * value;

  assert(not std::isnan(value));
  return value;
}

template <class ELEMENT_TYPE>
void UnbiasedSketch<ELEMENT_TYPE>::set(uint64_t i, ELEMENT_TYPE value) {
  uint64_t idx = findIndexInPhysicalVector(i);
  ELEMENT_TYPE& current_value = _physical_vector[idx];

  uint64_t sign_bit = hashFunction(i, _seed) % 2;
  current_value += sign_bit ? value : -1 * value;
}

template <class ELEMENT_TYPE>
void UnbiasedSketch<ELEMENT_TYPE>::assign(uint64_t size, ELEMENT_TYPE value) {
  (void)size;
  std::fill(_physical_vector.data(),
            _physical_vector.data() + _physical_vector.size(), value);
}

template <class ELEMENT_TYPE>
void UnbiasedSketch<ELEMENT_TYPE>::clear() {
  _physical_vector.clear();
}

template class UnbiasedSketch<float>;
template class BiasedSketch<float>;

}  // namespace thirdai::bolt
