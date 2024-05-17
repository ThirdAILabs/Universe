#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace thirdai::compression {

enum class CompressionScheme { Dragon = 0, CountSketch = 1 };

inline CompressionScheme convertStringToEnum(
    const std::string& compression_scheme) {
  std::string lower_name;
  for (char c : compression_scheme) {
    lower_name.push_back(std::tolower(c));
  }
  if (lower_name == "dragon") {
    return CompressionScheme::Dragon;
  }
  if (lower_name == "count_sketch") {
    return CompressionScheme::CountSketch;
  }
  throw std::invalid_argument("Invalid compression scheme specified.");
}

template <class T>
class CompressedVector {
 public:
  CompressedVector() {}

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

  virtual CompressionScheme type() const = 0;

  virtual uint32_t uncompressedSize() const = 0;

  virtual uint32_t size() const = 0;

  virtual ~CompressedVector() = default;

  /*
   * We pass a pointer to a char array to serialize function. The memory for
   * storing this array is allocated by the user before the serialize method is
   * called. This shifts the burden of managing the memory to the caller and
   * also makes it easier to work with memory leaks.
   */
  virtual void serialize(char* serialized_data) const = 0;

  virtual uint32_t serialized_size() const = 0;
};

}  // namespace thirdai::compression
