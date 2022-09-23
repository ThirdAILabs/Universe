#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>

namespace thirdai::compression {

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
               float compression_density, uint32_t seed_for_hashing,
               uint32_t sample_population_size);

  DragonVector(const T* values_to_compress, uint32_t size,
               float compression_density, uint32_t seed_for_hashing,
               uint32_t sample_population_size);

  explicit DragonVector(const char* serialized_data);

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  void clear() final;

  /*
   * Implementing utility methods for the class
   */

  void extend(const DragonVector<T>& vec);

  std::vector<uint32_t> indices() const { return _indices; }

  std::vector<T> values() const { return _values; }

  uint32_t seedForHashing() const { return _seed_for_hashing; }

  uint32_t uncompressedSize() const { return _uncompressed_size; }

  uint32_t size() const { return static_cast<uint32_t>(_indices.size()); }

  CompressionScheme type() const final;

  std::vector<T> decompress() const final;

  void serialize(char* serialized_data) const final;

  uint32_t serialized_size() const final;

 private:
  /*
   * If we add a lot of compression schemes, we should have a sparse vector
   * object rather than indices, values, size parameters. A lot of compression
   * schemes such as topk, randomk, dragon, dgc uses a sparse vector
   */

  std::vector<uint32_t> _indices;
  std::vector<T> _values;
  uint32_t _min_sketch_size = 10;

  // size of the original vector
  uint32_t _uncompressed_size;

  uint32_t _seed_for_hashing;

  void sketch(const T* values, T threshold, uint32_t size,
              uint32_t sketch_size);
};

}  // namespace thirdai::compression