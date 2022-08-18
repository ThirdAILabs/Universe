#pragma once

#include "CompressedVector.h"
#include <_types/_uint32_t.h>
#include <cstddef>
#include <cstdint>
#include <random>

namespace thirdai::compression {
template <class T>
class DefaultCompressedVector final : public CompressedVector<T> {
  // add a friend test class here

 public:
  DefaultCompressedVector<T>() {}

  explicit DefaultCompressedVector(const std::vector<T>& vec);

  explicit DefaultCompressedVector(const T* values, uint32_t size);

  /*
   * Implementing std::vector's standard methods for the class
   */

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  // we are only writing for a simple assign now, later expand to iterators and
  // array as well?
  void assign(uint32_t size, T value) final;

  void clear() final;

  /*
   * Implementing Operator methods for the class
   */

  DefaultCompressedVector<T> operator+(
      DefaultCompressedVector<T> const& vec) const;

  T operator[](uint32_t index) const final;

  /*
   * Implementing utility methods for the class
   */

  void extend(const DefaultCompressedVector<T>& vec);

  std::vector<DefaultCompressedVector<T>> split(size_t number_chunks) const;

  DefaultCompressedVector<T>& concat(DefaultCompressedVector<T> const& vec);

  bool isAllReducible() const final;

  std::vector<T> getValues() { return _values; }

  std::vector<T> decompressVector() const final;

 private:
  std::vector<T> _values;
  uint32_t _sketch_size;
};
}  // namespace thirdai::compression