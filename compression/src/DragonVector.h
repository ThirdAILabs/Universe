#pragma once

#include "CompressedVector.h"
#include <cstddef>
#include <cstdint>
#include <random>

namespace thirdai::compression {

// Interface for a DragonVector
template <class T>
class DragonVector final : public CompressedVector<T> {
 public:
  // defining the constructors for the class
  DragonVector<T>() {}

  /*
   * If we are constructing a dragon vector from (indices,values) then we need
   * to know the size of the original vector. Keeping track of the original size
   * is important when we want to decompress a vector.
   */
  DragonVector(const std::vector<T>& vector_to_compress,
               float compression_density, int seed_for_hashing);

  DragonVector(std::vector<uint32_t> indices, std::vector<T> values,
               uint32_t original_size, int seed_for_hashing);

  DragonVector(const T* values_to_compress, uint32_t size,
               float compression_density, int seed_for_hashing);

  /*
   * Implementing std::vector's standard methods for the class
   */

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  // we are only writing for a simple assign now, later expand to iterators and
  // array as well?
  void assign(uint32_t size, T value);

  void assign(uint32_t size, uint32_t index, T value,
              uint32_t original_size = 0);

  void clear() final;

  /*
   * Implementing utility methods for the class
   */

  void extend(const DragonVector<T>& vec);

  /*
   * Dragon vectors are not additive by default. But we can still define schemes
   * to add them up.
   */

  bool isAdditive() const final;

  std::vector<uint32_t> getIndices() { return _indices; }

  std::vector<T> getValues() { return _values; }

  int getSeedForHashing() const { return _seed_for_hashing; }

  uint32_t getOriginalSize() const { return _original_size; }

  float getCompressionDensity() const { return _compression_density; }

  uint32_t size() const { return static_cast<uint32_t>(_indices.size()); }

  std::string getCompressionScheme() const final;

  /*
   * We are storing indices,values tuple hence, decompressing is just
   * putting corresponding values for the stored indices
   */
  std::vector<T> decompress() const final;

 private:
  /*
   * If we add a lot of compression schemes, we should have a sparse vector
   * object rather than indices, values, size parameters. A lot of compression
   * schemes such as topk, randomk, dragon, dgc uses a sparse vector
   */

  std::vector<uint32_t> _indices;
  std::vector<T> _values;
  uint32_t _min_sketch_size = 10;
  uint32_t _original_size = 0;
  float _compression_density = 1;
  int _seed_for_hashing;

  void sketchVector(const T* values, T threshold, uint32_t size,
                    uint32_t sketch_size);
};
}  // namespace thirdai::compression

/*
 * Exception handling?
 * While getting element at a location, check if element is present at the
 * hashed location in the indices array. Size exceptions for the methods. Not
 * adding two vectors with different seed_for_hashing? Should we make this
 * sketch all reducible? or just concatenation? minimum size of the sketch
 */