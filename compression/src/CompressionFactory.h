#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include "CountSketch.h"
#include "DragonVector.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <stdexcept>
#include <variant>
namespace thirdai::compression {

/*
 * We are using visitor patterns to deal with runtime polymorphism of compressed
 * vector objects. Using visitor pattern also makes sure that we don't have to
 * change the object implementations whenever we want to add a new feature.
 * For example, we have a binary operation called extend that operates on two
 * compressed vectors of the same type. Adding two compressed vector of
 * different classes should throw an error. Using visitor pattern, we can handle
 * these exceptions without modifying the existing classes.
 */
template <class T>
class ExtendVisitor {
 public:
  void operator()(DragonVector<T>& vector_to_extend,
                  const DragonVector<T>& vector_to_extend_with) {
    vector_to_extend.extend(vector_to_extend_with);
  }
  void operator()(CountSketch<T>& vector_to_extend,
                  const CountSketch<T>& vector_to_extend_with) {
    vector_to_extend.extend(vector_to_extend_with);
  }
  void operator()(DragonVector<T>& vector_to_extend,
                  const CountSketch<T>& vector_to_extend_with) {
    (void)vector_to_extend;
    (void)vector_to_extend_with;
    throw std::invalid_argument(
        "Cannot extend a DragonVector with a CountSketch");
  }
  void operator()(CountSketch<T>& vector_to_extend,
                  const DragonVector<T>& vector_to_extend_with) {
    (void)vector_to_extend;
    (void)vector_to_extend_with;
    throw std::invalid_argument(
        "Cannot extend a CountSketch with a DragonVector");
  }
};

template <class T>
class DecompressVisitor {
 public:
  std::vector<T> operator()(const DragonVector<T>& dragon_vector) {
    return dragon_vector.decompress();
  }
  std::vector<T> operator()(const CountSketch<T>& count_sketch) {
    return count_sketch.decompress();
  }
};

template <class T>
class SizeVisitor {
 public:
  uint32_t operator()(const DragonVector<T>& dragon_vector) {
    return dragon_vector.serialized_size();
  }
  uint32_t operator()(const CountSketch<T>& count_sketch) {
    return count_sketch.serialized_size();
  }
};

template <class T>
class SerializeVisitor {
 public:
  explicit SerializeVisitor<T>(char* pointer_to_write_to)
      : _serialized_data(pointer_to_write_to) {}

  void operator()(const DragonVector<T>& dragon_vector) {
    dragon_vector.serialize(_serialized_data);
  }
  void operator()(const CountSketch<T>& count_sketch) {
    count_sketch.serialize(_serialized_data);
  }

 private:
  char* _serialized_data;
};

template <class T>
inline std::variant<DragonVector<T>, CountSketch<T>> compress(
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
inline std::variant<DragonVector<T>, CountSketch<T>> concat(
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
}  // namespace thirdai::compression
