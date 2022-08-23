#pragma once

#include "CompressionUtils.h"
#include <cstdint>
#include <memory>
#include <random>

namespace thirdai::compression {
enum class CompressionScheme {
  CountMin,
  CountSketch,
  Dragon,
  UnbiasedDragon,
  Default
};

// a generic compressed vector class
template <class T>
class CompressedVector {
 public:
  CompressedVector<T>() {}

  // std::vector methods for compressed vector

  virtual T get(uint32_t index) const = 0;

  virtual void set(uint32_t index, T value) = 0;

  virtual void assign(uint32_t size, T value) = 0;

  virtual void clear() = 0;

  CompressedVector<T> operator+(CompressedVector<T> const& vec);

  virtual T operator[](uint32_t index) const = 0;

  // methods for the compressed_vector class

  virtual bool isAdditive() const = 0;

  void extend(const CompressedVector<T>& vec);

  virtual std::vector<T> decompressVector() const = 0;

  virtual ~CompressedVector() = default;
};
}  // namespace thirdai::compression

/*
 * We should also create a default compressed vector class that is exactly like
 * std::vector but has other functionalities like extend, split, concat etc.
 * We will have scenarios where we only want to compress gradients or parameters
 * for certain layers and not the other ones, but at the same time, we do not
 * want distributed bolt to be bothered with what vectors are compressed and
 * what are normal. Hence, should be some default compressed vector inheriting
 * from the vector class. And implementing all these functions.
 */

/*
 * We will also need to have objects like sparse vector, dense vector.
 */