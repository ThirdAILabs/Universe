#pragma once

#include <hashing/src/UniversalHash.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
namespace thirdai::compression {

/*
 * Rather than getting an exact threshold, we sample a number of points from the
 * values array and gets an estimate for topk.
 * If threshold_ratio = 0.1 => estimateTopKThreshold will return a value which
 * is larger than 90% of the values.
 */
template <class T>
inline T estimateTopKThreshold(const T* values, uint32_t size,
                               float threshold_ratio,
                               uint32_t seed_for_sampling,
                               uint32_t sample_population_size) {
  // sample_population_size is the total number of random samples we take for
  // estimating a threshold for the values

  uint32_t num_top_k =
      static_cast<uint32_t>(sample_population_size * threshold_ratio);

  std::vector<T> sampled_values(sample_population_size, 0);

  std::mt19937 gen(seed_for_sampling);
  std::uniform_int_distribution<> distribution(0, size - 1);
  for (uint32_t i = 0; i < sample_population_size; i++) {
    sampled_values[i] = std::abs(values[distribution(gen)]);
  }
  /*
   * This is Quickselect. We can find the i'th largest element using
   * nth_element, which is exactly what we need to do now. Works on unsorted
   * arrays too.
   */
  std::nth_element(sampled_values.begin(),
                   sampled_values.begin() + sample_population_size - num_top_k,
                   sampled_values.end());

  // threshold is an estimate for the kth largest element in the gradients
  // matrix
  T estimated_threshold = sampled_values[sample_population_size - num_top_k];
  return estimated_threshold;
}

}  // namespace thirdai::compression