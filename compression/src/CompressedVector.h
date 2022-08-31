#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
namespace thirdai::compression {

// a generic compressed vector class
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
   * Count sketches, count-min sketches are additive in nature. Others are not.
   * All the derived compression schemes should implement this function so that
   * we do not add two non-additive count sketches.
   */
  virtual bool isAdditive() const = 0;

  /*
   * Extending a sketch is appending the given sketch to the current object.
   * Similar to additiveness, not all sketches are extendible for e.g.,
   * count-sketches.
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

// DragonVector class
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
  uint32_t _original_size = 0;
  float _compression_density = 1;
  int _seed_for_hashing;

  void sketchVector(const T* values, T threshold, uint32_t size,
                    uint32_t sketch_size);
};

template <class T>
inline std::unique_ptr<CompressedVector<T>> compress(
    const std::vector<T>& values, const std::string& compression_scheme = "",
    float compression_density = 1, int seed_for_hashing = 0) {
  return compress(values.data(), static_cast<uint32_t>(values.size()),
                  compression_scheme, compression_density, seed_for_hashing);
}

template <class T>
inline std::unique_ptr<CompressedVector<T>> compress(
    const T* values, uint32_t size, const std::string& compression_scheme = "",
    float compression_density = 1, int seed_for_hashing = 0) {
  if (compression_scheme == "dragon") {
    return std::make_unique<DragonVector<T>>(values, size, compression_density,
                                             seed_for_hashing);
  }
  throw std::logic_error("Compression Scheme is invalid");
}

template <class T>
inline std::vector<T> decompress(const CompressedVector<T>& compressed_vector) {
  return compressed_vector.decompress();
}

}  // namespace thirdai::compression
