#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>

namespace thirdai::compression {

template <class T>
class CompressedVector {
 public:
  CompressedVector<T>() {}

  /*
   * Index refers to the index in the uncompressed_vector.
   * Since, the vector is compressed, the returned value is just an estimate and
   * not exact.
   */
  virtual T get(uint32_t index) const = 0;

  virtual void set(uint32_t index, T value) = 0;

  virtual void clear() = 0;

  // methods for the compressed_vector class

  /*
   * Extending a sketch is appending the given sketch to the current object.
   * Each compressed vector type will have its own logic for extending.
   * Extending a CompressedVector by another is supposed to be non-lossy in
   * nature as all the data from the two vectors is simply concatenated. In
   * contrast, add might be lossy in nature but is supposed to be all-reducible.
   * Hence, there is a tradeoff:
   *  extend: non-lossy but more memory
   *  add: lossy but memory footprint does not change.
   */
  void extend(const CompressedVector<T>& vec);

  /*
   * Returns a std::vector formed by decompressing the compressed vector. This
   * method should be implemented by all the schemes.
   */
  virtual std::vector<T> decompress() const = 0;

  virtual std::string type() const = 0;

  virtual ~CompressedVector() = default;

  virtual std::stringstream serialize() const = 0;

  virtual uint32_t serialized_size() const = 0;
};

}  // namespace thirdai::compression
