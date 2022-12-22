#include "CompressionFactory.h"
#include <compression/src/DragonVector.h>
#include <stdexcept>
#include <variant>

namespace thirdai::compression {
template <class T>
std::variant<DragonVector<T>, CountSketch<T>> compress(
    const T* values, uint32_t size, const std::string& compression_scheme,
    float compression_density, uint32_t seed_for_hashing,
    uint32_t sample_population_size) {
  CompressionScheme compression_scheme_enum =
      convertStringToEnum(compression_scheme);

  switch (compression_scheme_enum) {
    case CompressionScheme::Dragon:
      return DragonVector<T>(values, size, compression_density,
                             seed_for_hashing, sample_population_size);

    case CompressionScheme::CountSketch: {
      /*
       * Count sketches is a stack of multiple sketches and requires a seed for
       * each of the sketches. Rather than asking the caller to give seeds for
       * all the sketches, we get the seeds for the sketches by incrementing the
       * input seed.
       */
      uint32_t num_sketches = sample_population_size;
      std::vector<uint32_t> seed_for_hashing_indices;
      std::vector<uint32_t> seed_for_sign;

      for (uint32_t i = 0; i < num_sketches; i++) {
        seed_for_hashing_indices.push_back(i + seed_for_hashing);
        seed_for_sign.push_back(i + seed_for_hashing);
      }

      return CountSketch<T>(values, size, compression_density, num_sketches,
                            seed_for_hashing_indices, seed_for_sign);
    }
  }
  throw std::invalid_argument(
      "Compression Scheme not supported. Only supports dragon and "
      "count_sketch");
}

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> concat(
    std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
        compressed_vectors) {
  if (compressed_vectors.empty()) {
    throw std::logic_error("No compressed vectors provided for concatenating.");
  }
  // We initialize a compressed vector from the first element of the input
  // vector, and then keep on extending it with the rest of the elements.

  std::variant<DragonVector<T>, CountSketch<T>> initial_compressed_vector(
      compressed_vectors[0]);
  size_t num_vectors = compressed_vectors.size();
  for (size_t i = 1; i < num_vectors; i++) {
    std::visit(ExtendVisitor<T>(), initial_compressed_vector,
               compressed_vectors[i]);
  }
  return initial_compressed_vector;
}

template <class T>
std::variant<DragonVector<T>, CountSketch<T>> add(
    std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
        compressed_vectors) {
  if (compressed_vectors.empty()) {
    throw std::logic_error("No compressed vectors provided for addition.");
  }
  // We initialize a compressed vector from the first element of the input
  // vector, and then keep on extending it with the rest of the elements.
  std::variant<DragonVector<T>, CountSketch<T>> initial_compressed_vector(
      compressed_vectors[0]);
  size_t num_vectors = compressed_vectors.size();
  for (size_t i = 1; i < num_vectors; i++) {
    std::visit(AddVisitor<T>(), initial_compressed_vector,
               compressed_vectors[i]);
  }
  std::visit(DivideVisitor<T>(static_cast<uint32_t>(compressed_vectors.size())),
             initial_compressed_vector);
  return initial_compressed_vector;
}

template std::variant<DragonVector<float>, CountSketch<float>> add(
    std::vector<std::variant<DragonVector<float>, CountSketch<float>>>);

template std::variant<DragonVector<float>, CountSketch<float>> concat(
    std::vector<std::variant<DragonVector<float>, CountSketch<float>>>);

template std::variant<DragonVector<float>, CountSketch<float>> compress(
    const float* values, uint32_t size, const std::string& compression_scheme,
    float compression_density, uint32_t seed_for_hashing,
    uint32_t sample_population_size);
}  // namespace thirdai::compression