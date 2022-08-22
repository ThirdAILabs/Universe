#pragma once

#include "CompressedVector.h"
#include <_types/_uint32_t.h>
#include <cstddef>
#include <cstdint>
#include <random>

namespace thirdai::compression {

template <class T>
class DragonVector final : public CompressedVector<T> {
  // add a friend test class here

 public:
  DragonVector<T>() {}

  // might have to remove these =delete and declare explicit copy constructors
  // by ourselves

  DragonVector(const DragonVector<T>& vec);
  // DragonVector(const DragonVector<T>&) = delete;
  // DragonVector(DragonVector<T>&&) = delete;

  DragonVector& operator=(DragonVector<T> vec) {
    swap(*this, vec);
    return *this;
  }

  DragonVector(DragonVector<T>&& vec) : DragonVector<T>() { swap(*this, vec); }

  friend void swap(DragonVector<T>& first, DragonVector<T>& second) {
    using std::swap;
    swap(first._indices, second._indices);
    swap(first._values, second._values);
    swap(first._min_sketch_size, second._min_sketch_size);
    swap(first._original_size, second._original_size);
    swap(first._sketch_size, second._sketch_size);
    swap(first._compression_density, second._compression_density);
    swap(first._seed_for_hashing, second._seed_for_hashing);
  }

  // DragonVector& operator=(DragonVector<T>&&) = delete;

  // defining the constructors for the class

  DragonVector(const std::vector<T>& vec, float compression_density,
               int seed_for_hashing);

  DragonVector(std::vector<uint32_t> indices, std::vector<T> values,
               uint32_t size, uint32_t original_size, int seed_for_hashing);

  DragonVector(const T* values, float compression_density, uint32_t size,
               int seed_for_hashing);

  /*
   * Implementing std::vector's standard methods for the class
   */

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  // we are only writing for a simple assign now, later expand to iterators and
  // array as well?
  void assign(uint32_t size, T value) final;

  void assign(uint32_t size, uint32_t index, T value,
              uint32_t original_size = 0);

  void clear() final;

  /*
   * Implementing Operator methods for the class
   */

  DragonVector<T> operator+(DragonVector<T> const& vec) const;

  T operator[](uint32_t index) const final;

  /*
   * Implementing utility methods for the class
   */

  void extend(const DragonVector<T>& vec);

  std::vector<DragonVector<T>> split(size_t number_chunks) const;

  DragonVector<T>& concat(DragonVector<T> const& vec);

  bool isAllReducible() const final;

  std::vector<uint32_t> getIndices() { return _indices; }

  std::vector<T> getValues() { return _values; }

  int getSeedForHashing() const { return _seed_for_hashing; }

  uint32_t getOriginalSize() const { return _original_size; }

  float getCompressionDensity() const { return _compression_density; }

  uint32_t getSketchSize() const { return _sketch_size; }

  std::vector<T> decompressVector() const final;

 private:
  /*
   * If we add a lot of compression schemes, we should have a sparse vector
   * object rather than indices, values, size parameters. A lot of compression
   * schemes such as topk, randomk, dragon, dgc uses a sparse vector
   */

  std::vector<uint32_t> _indices;
  std::vector<T> _values;
  uint32_t _min_sketch_size = 10;
  uint32_t _sketch_size = 0;
  uint32_t _original_size = 0;
  float _compression_density = 1;
  int _seed_for_hashing;

  void sketchVector(const std::vector<T>& vec, T threshold);

  void sketchVector(const T* values, T threshold, uint32_t size);
};
}  // namespace thirdai::compression

/*
 * Exception handling?
 * While getting element at a location, check if element is present at the
 * hashed location in the indices array. Size exceptions for the methods. Not
 * adding two vectors with different seed_for_hashing? Should we make this
 * sketch all reducible? or just concatenation? minimum size of the sketch
 */