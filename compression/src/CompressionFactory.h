#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include "CountSketch.h"
#include "DragonVector.h"
#include <compression/src/CompressionUtils.h>
#include <memory>
#include <stdexcept>

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
inline std::vector<T> decompress(const CompressedVector<T>& compressed_vector) {
  return compressed_vector.decompress();
}

template <class T>
inline std::unique_ptr<CompressedVector<T>> concat(
    std::vector<std::unique_ptr<CompressedVector<T>>> compressed_vectors) {
  // We take the compressed vector at the 0th index and concatenate all other
  // compressed vectors to it.

  std::unique_ptr<CompressedVector<T>> initial_vector(
      std::move(compressed_vectors[0]));

  for (size_t i = 1; i < compressed_vectors.size(); i++) {
    initial_vector->extend(std::move(compressed_vectors[i]));
  }

  return initial_vector;
}
}  // namespace thirdai::compression
