#pragma once

#include "CompressedVector.h"
#include <cstddef>
#include <cstdint>
#include <random>

namespace thirdai::compression {
template <class T>
class DefaultCompressedVector final : public CompressedVector<T> {
  // add a friend test class here

 public:
  // defining the constructors for the class

  DefaultCompressedVector<T>() {}

  explicit DefaultCompressedVector(const std::vector<T>& vec);

  DefaultCompressedVector(const T* values, uint32_t size);

  DefaultCompressedVector(const DefaultCompressedVector<T>& vec);

  explicit DefaultCompressedVector(std::vector<T>&& vec);

  // using "copy-swap idiom" for = operator. This implementation makes sure that
  // we do not have to check for self-reference.
  DefaultCompressedVector& operator=(DefaultCompressedVector<T> vec) {
    swap(*this, vec);
    return *this;
  }

  DefaultCompressedVector(DefaultCompressedVector<T>&& vec)
      : DefaultCompressedVector<T>() {
    swap(*this, vec);
  }

  friend void swap(DefaultCompressedVector<T>& first,
                   DefaultCompressedVector<T>& second) {
    using std::swap;
    swap(first._values, second._values);
  }

  /*
   * Implementing std::vector's standard methods for the class
   */

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  // we are only writing for a simple assign now, later expand to iterators and
  // array as well?
  void assign(uint32_t size, T value);

  void clear() final;

  /*
   * Implementing Operator methods for the class
   */

  DefaultCompressedVector<T> operator+(
      DefaultCompressedVector<T> const& vec) const;

  /*
   * To-Do(Shubh):
   * This method should return a reference to the element at the index so that
   * we can do things like vector[i]=a.
   */

  T operator[](uint32_t index) const final;

  /*
   * Implementing utility methods for the class
   */

  void extend(const DefaultCompressedVector<T>& vec);

  std::vector<DefaultCompressedVector<T>> split(size_t number_chunks) const;

  bool isAdditive() const final;

  std::vector<T> getValues() { return _values; }

  std::vector<T> decompressVector() const final;

  uint32_t size() { return _values.size(); }

  std::string getCompressionScheme() const final { return "default"; }

 private:
  std::vector<T> _values;
};
}  // namespace thirdai::compression