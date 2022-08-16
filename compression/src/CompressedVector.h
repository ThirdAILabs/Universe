#pragma once

#include "CompressionUtils.h"
#include <_types/_uint32_t.h>
#include <cstdint>
#include <random>

namespace thirdai::compression {
enum class CompressionScheme {
  CountMin,
  CountSketch,
  Dragon,
  UnbiasedDragon,
  Default
};

template <class T>

class CompressedVector {
 public:
  explicit CompressedVector<T>(std::string compression_scheme,
                               const std::vector<T>& vec,
                               float compression_density);

  explicit CompressedVector<T>(std::string compression_scheme);

  virtual bool isAllReducible() = 0;

  virtual std::unique_ptr<CompressedVector> extend(
      const std::vector<T>& raw) = 0;

  virtual std::unique_ptr<std::vector<CompressedVector<T>>> split(
      int number_chunks) = 0;

  // std::vector methods for compressed vector

  virtual T operator[](uint32_t index) = 0;

  virtual T get(uint32_t index) const = 0;

  virtual void set(uint32_t index, T value) = 0;

  virtual void assign(uint32_t size, T value) = 0;

  // write more methods for addition, subtraction, multiplying by -1, union,
  // etc.

  virtual CompressedVector<T> operator+(CompressedVector<T> const& obj) = 0;

  virtual void clear() = 0;

  virtual ~CompressedVector() = default;

 protected:
  CompressionScheme _compression_scheme;
};
}  // namespace thirdai::compression