#include "CompressedVector.h"
#include <iostream>

namespace thirdai::bolt {

namespace {

template <class ELEMENT_TYPE>
void debugCompressed(const std::vector<ELEMENT_TYPE>& original,
                     CompressedVector<ELEMENT_TYPE>* compressed) {
  for (size_t i = 0; i < original.size(); i++) {
    if (original[i] != compressed->get(i)) {
      std::cout << i << ": " << original[i] << ", " << compressed->get(i)
                << "\n";
    }
  }
}

}  // namespace

// Create a new BiasedSketch.
template <class ELEMENT_TYPE>
BiasedSketch<ELEMENT_TYPE>::BiasedSketch(
    uint64_t physical_size, ELEMENT_TYPE default_value /* = 0*/,
    uint64_t block_size /* = kDefaultBlockSize*/,
    uint32_t seed /*= kDefaultSeed*/)
    : _sketch(physical_size + block_size, default_value),
      _util(seed, block_size, physical_size) {}

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

  for (uint64_t i = 0; i < input.size(); i += _util.block_size()) {
    // Find the location the first element of the block hashes into.
    // Hashing is truncated by truncated_size to avoid out of bounds access in
    // the nested loop below.

    uint64_t block_begin = _util.hash(i) % _util.container_size();

    // Having found the hash, we store all elements in the block within the
    // respective offset.
    for (uint64_t j = i; j < i + _util.block_size(); j++) {
      uint64_t offset = j - i;
      uint64_t index = block_begin + offset;

      _sketch[index] += input[j];
    }
  }
}

// non-const accessor.
template <class ELEMENT_TYPE>
ELEMENT_TYPE BiasedSketch<ELEMENT_TYPE>::get(uint64_t i) const {
  uint64_t idx = _util.projectedIndex(i);
  ELEMENT_TYPE value = _sketch[idx];
  return value;
}

// Set a value at an index.
template <class ELEMENT_TYPE>
void BiasedSketch<ELEMENT_TYPE>::set(uint64_t i, ELEMENT_TYPE value) {
  uint64_t idx = _util.projectedIndex(i);
  ELEMENT_TYPE& current_value = _sketch[idx];

  // @jerin-thirdai was supposed to use the following (aggregate without sign).
  // Replacing the value only appears to work, the other generates a lot of
  // NaNs.
  //
  // current_value += value;
  current_value = value;
}

template <class ELEMENT_TYPE>
void BiasedSketch<ELEMENT_TYPE>::assign(uint64_t size, ELEMENT_TYPE value) {
  (void)size;
  std::fill(_sketch.data(), _sketch.data() + _sketch.size(), value);
}

template <class ELEMENT_TYPE>
void BiasedSketch<ELEMENT_TYPE>::clear() {
  _sketch.clear();
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

  for (uint64_t i = 0; i < input.size(); i += _util.block_size()) {
    // Find the location the first element of the block hashes into.
    // Hashing is truncated by truncated_size to avoid out of bounds access in
    // the nested loop below.

    uint64_t block_begin = _util.hash(i) % _util.container_size();

    // Having found the hash, we store all elements in the block within the
    // respective offset.
    for (uint64_t j = i; j < i + _util.block_size(); j++) {
      uint64_t offset = j - i;
      uint64_t index = block_begin + offset;

      bool sign_bit = _util.hash(j) % 2;

      // Add the input value multiplied by sign bit to the index at
      // _sketch.
      if (sign_bit) {
        _sketch[index] += input[j];
      } else {
        _sketch[index] -= input[j];
      }
    }
  }
}

template <class ELEMENT_TYPE>
UnbiasedSketch<ELEMENT_TYPE>::UnbiasedSketch(
    uint64_t physical_size, ELEMENT_TYPE default_value /* = 0*/,
    uint64_t block_size /* = kDefaultBlockSize*/,
    uint32_t seed /* = kDefaultSeed*/)
    : _sketch(physical_size + block_size, default_value),
      _util(seed, block_size, physical_size) {}

template <class ELEMENT_TYPE>
ELEMENT_TYPE UnbiasedSketch<ELEMENT_TYPE>::get(uint64_t i) const {
  uint64_t idx = _util.projectedIndex(i);
  ELEMENT_TYPE value = _sketch[idx];

  uint64_t sign_bit = _util.hash(i) % 2;
  value = sign_bit ? value : -1 * value;

  assert(not std::isnan(value));
  return value;
}
template <class ELEMENT_TYPE>
void UnbiasedSketch<ELEMENT_TYPE>::set(uint64_t i, ELEMENT_TYPE value) {
  uint64_t idx = _util.projectedIndex(i);
  ELEMENT_TYPE& current_value = _sketch[idx];

  uint64_t sign_bit = _util.hash(i) % 2;
  current_value += sign_bit ? value : -1 * value;
}

template <class ELEMENT_TYPE>
void UnbiasedSketch<ELEMENT_TYPE>::assign(uint64_t size, ELEMENT_TYPE value) {
  (void)size;
  std::fill(_sketch.data(), _sketch.data() + _sketch.size(), value);
}

template <class ELEMENT_TYPE>
void UnbiasedSketch<ELEMENT_TYPE>::clear() {
  _sketch.clear();
}

template class UnbiasedSketch<float>;
template class BiasedSketch<float>;

}  // namespace thirdai::bolt
