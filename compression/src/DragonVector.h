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

  // DragonVector(const DragonVector<T>&) = delete;
  // DragonVector(DragonVector<T>&&) = delete;
  // DragonVector& operator=(const DragonVector<T>&) = delete;
  // DragonVector& operator=(DragonVector<T>&&) = delete;

  // defining the constructors for the class

  explicit DragonVector(const std::vector<T>& vec, float compression_density,
                        int seed_for_hashing);

  explicit DragonVector(std::vector<uint32_t> indices, std::vector<T> values,
                        uint32_t size, int seed_for_hashing);

  explicit DragonVector(const T* values, float compression_density,
                        uint32_t size, int seed_for_hashing);

  // compatibility functions with std::vector

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  // we are only writing for a simple assign now, later expand to iterators and
  // array as well?
  void assign(uint32_t size, T value) final;

  void assign(uint32_t size, uint32_t index, T value);

  void clear() final;

  // write more methods for addition, subtraction, multiplying by -1, union,
  // etc.

  DragonVector<T> operator+(DragonVector<T> const& vec) const;

  T operator[](uint32_t index) const final;

  // methods for the Dragon Vector Class
  void extend(const DragonVector<T>& vec);

  std::vector<DragonVector<T>> split(size_t number_chunks) const;

  DragonVector<T>& concat(DragonVector<T> const& vec);

  bool isAllReducible() const final;

  std::vector<uint32_t> getIndices() { return _indices; }

  std::vector<T> getValues() { return _values; }

 private:
  /*
   * If we add a lot of compression schemes, we should have a sparse vector
   * object rather than indices, values, size parameters. A lot of compression
   * schemes such as topk, randomk, dragon, dgc uses a sparse vector
   */

  std::vector<uint32_t> _indices;
  std::vector<T> _values;
  uint32_t _sketch_size;
  float _compression_density = 1;
  int _seed_for_hashing;
  uint32_t _min_sketch_size = 10;

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