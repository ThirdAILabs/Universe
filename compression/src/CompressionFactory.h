#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include "CountSketch.h"
#include "DragonVector.h"
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <variant>
namespace thirdai::compression {

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
inline std::variant<DragonVector<T>, CountSketch<T>> compressVariant(
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
}

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
  if (compression_scheme == "count_sketch") {
    /*
     * Count sketches is a stack of multiple sketches and requires a seed for
     * each of the sketches. Rather than asking the caller to give seeds for all
     * the sketches, we get the seeds for the sketches by incrementing the input
     * seed.
     */
    uint32_t num_sketches = sample_population_size;
    std::vector<uint32_t> seed_for_hashing_indices;
    std::vector<uint32_t> seed_for_sign;

    for (uint32_t i = 0; i < num_sketches; i++) {
      seed_for_hashing_indices.push_back(i + seed_for_hashing);
      seed_for_sign.push_back(i + seed_for_hashing);
    }

    return std::make_unique<CountSketch<T>>(
        values, size, compression_density, num_sketches,
        seed_for_hashing_indices, seed_for_sign);
  }
  throw std::logic_error(
      "The provided compression scheme is invalid. The compression module "
      "supports dragon and count_sketch.");
}

template <class T>
inline DragonVector<T> concat(
    const std::vector<DragonVector<T>>& compressed_dragon_vectors) {
  DragonVector<T> initial_dragon_vector =
      compressed_dragon_vectors[0];  // a copy being made of 0th vector

  for (size_t i = 1; i < compressed_dragon_vectors.size(); i++) {
    initial_dragon_vector.extend(compressed_dragon_vectors[i]);
  }

  return std::move(initial_dragon_vector);
}

template <class T>
inline CountSketch<T> concat(
    const std::vector<CountSketch<T>>& compressed_count_sketches) {
  CountSketch<T> initial_count_sketch =
      compressed_count_sketches[0];  // a copy being made of 0th vector

  for (size_t i = 1; i < compressed_count_sketches.size(); i++) {
    initial_count_sketch.extend(compressed_count_sketches[i]);
  }

  return std::move(initial_count_sketch);
}

template <class T>
inline std::variant<DragonVector<T>, CountSketch<T>> concatVariant(
    std::vector<std::variant<DragonVector<T>, CountSketch<T>>>
        compressed_vectors) {
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
