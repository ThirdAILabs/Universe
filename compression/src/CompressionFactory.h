#include "CompressedVector.h"
#include "CompressionUtils.h"
#include "DragonVector.h"
#include <compression/src/CompressionUtils.h>

namespace thirdai::compression {
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
