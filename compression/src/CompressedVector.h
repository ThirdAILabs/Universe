#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>

namespace thirdai::compression {

template <class T>
class CompressedVector {
 public:
  CompressedVector<T>() {}

  // std::vector methods for compressed vector

  virtual T get(uint32_t index) const = 0;

  virtual void set(uint32_t index, T value) = 0;

  virtual void clear() = 0;

  // methods for the compressed_vector class

  /*
   * Extending a sketch is appending the given sketch to the current object.
   * Each compressed vector type will have its own logic for extending
   */
  void extend(const CompressedVector<T>& vec);

  /*
   * Returns a std::vector formed by decompressing the compressed vector. This
   * method should be implemented by all the schemes.
   */
  virtual std::vector<T> decompress() const = 0;

  virtual std::string type() const = 0;

  virtual ~CompressedVector() = default;
};

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

  DragonVector(std::vector<uint32_t> indices, std::vector<T> values,
               uint32_t uncompressed_size, uint32_t seed_for_hashing);

  DragonVector(const T* values_to_compress, uint32_t size,
               float compression_density, uint32_t seed_for_hashing,
               uint32_t sample_population_size);

  /*
   * Implementing std::vector's standard methods for the class
   */

  // index refers to the index in the uncompressed_vector

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  void clear() final;

  /*
   * Implementing utility methods for the class
   */

  /*
   * Extending a DragonVector by another is simply concatenating the indices and
   * values vectors of the two. Extend is non-lossy in nature as all the hashed
   * indices and values pairs are present. In contrast, add is lossy in nature
   * but is all-reducible.
   * Hence, there is a tradeoff:
   *  extend: non-lossy but more memory
   *  add: lossy but memory footprint does not change.
   */
  void extend(const DragonVector<T>& vec);

  std::vector<uint32_t> indices() { return _indices; }

  std::vector<T> values() { return _values; }

  int seedForHashing() const { return _seed_for_hashing; }

  uint32_t uncompressedSize() const { return _uncompressed_size; }

  float compressionDensity() const { return _compression_density; }

  uint32_t size() const { return static_cast<uint32_t>(_indices.size()); }

  std::string type() const final;

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

  // size of the original vector
  uint32_t _uncompressed_size;

  float _compression_density;
  int _seed_for_hashing;

  void sketch(const T* values, T threshold, uint32_t size,
              uint32_t sketch_size);
};

template <class T>
inline std::unique_ptr<CompressedVector<T>> compress(
    const T* values, uint32_t size, const std::string& compression_scheme,
    float compression_density, uint32_t seed_for_hashing,
    uint32_t sample_population_size) {
  if (compression_scheme == "dragon") {
    return std::make_unique<DragonVector<T>>(values, size, compression_density,
                                             seed_for_hashing,
                                             sample_population_size);
  }
  throw std::logic_error("Compression Scheme is invalid");
}

template <class T>
inline std::vector<T> decompress(const CompressedVector<T>& compressed_vector) {
  return compressed_vector.decompress();
}

}  // namespace thirdai::compression
