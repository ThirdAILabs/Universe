#pragma once

#include <hashing/src/UniversalHash.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace thirdai::compression {

/*
 * Rather than getting an exact threshold, we sample a number of points from the
 * values array and gets an estimate for topk.
 * If top_k = 0.1 => threshold will return a value which is larger than 90% of
 * the values.
 */
template <class T>
inline T thresholdForTopK(const T* values, uint32_t size, float top_k,
                          uint32_t seed_for_sampling,
                          uint32_t sample_population_size) {
  // sample_population_size is the total number of random samples we take for
  // estimating a threshold for the values

  uint32_t top_k_index = static_cast<uint32_t>(sample_population_size * top_k);

  std::vector<T> sampled_values(sample_population_size, 0);

  std::mt19937 gen(seed_for_sampling);
  std::uniform_int_distribution<> distribution(0, size - 1);
  for (uint32_t i = 0; i < sample_population_size; i++) {
    sampled_values[i] = std::abs(values[distribution(gen)]);
  }

  std::nth_element(
      sampled_values.begin(),
      sampled_values.begin() + sample_population_size - top_k_index,
      sampled_values.end());

  // threshold is an estimate for the kth largest element in the gradients
  // matrix
  T threshold = sampled_values[sample_population_size - top_k_index];
  return threshold;
}

}  // namespace thirdai::compression