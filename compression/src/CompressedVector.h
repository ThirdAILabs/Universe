#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressionUtils.h>
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

  void assign(uint32_t size, T value);

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

  virtual std::string getCompressionScheme() const = 0;

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