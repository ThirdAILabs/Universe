#pragma once

#include <hashing/src/UniversalHash.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace thirdai::compression {

// To-Do:
// Remove printVector function before merging the PR
template <class T>
inline void printVector(std::vector<T> vec, const std::string& name) {
  std::cout << "Printing the vector " << name << std::endl;
  for (auto x : vec) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

// an approximation for top-k threshold by random sampling
template <class T>
inline T getThresholdForTopK(const std::vector<T>& values, uint32_t sketch_size,
                             uint32_t max_samples_for_random_sampling,
                             int seed_for_sampling) {
  return getThresholdForTopK(values.data, static_cast<uint32_t>(values.size()),
                             sketch_size, max_samples_for_random_sampling,
                             seed_for_sampling);
}

/*
 * Rather than getting an exact threshold, we sample a number of points from the
 * values array and gets an estimate for topk.
 */
template <class T>
inline T getThresholdForTopK(const T* values, uint32_t size,
                             uint32_t sketch_size,
                             uint32_t max_samples_for_random_sampling,
                             int seed_for_sampling) {
  uint32_t num_samples = std::min(max_samples_for_random_sampling, sketch_size);

  uint32_t min_samples = 20;

  /*
   * There will be scenarios when num_samples takes very small values such as 1,
   * hence, we ensure that we sample at least 20 points.
   */

  if (num_samples < min_samples) {
    num_samples = static_cast<uint32_t>(std::min(min_samples, size));
  }

  float compression_vector = static_cast<float>(sketch_size) / size;
  uint32_t topK = static_cast<uint32_t>(num_samples * compression_vector);

  std::vector<T> sampled_gradients(num_samples, 0);

  std::mt19937 gen(seed_for_sampling);
  std::uniform_int_distribution<> distrib(0, size - 1);
  for (uint32_t i = 0; i < num_samples; i++) {
    sampled_gradients[i] = std::abs(values[distrib(gen)]);
  }

  std::nth_element(sampled_gradients.begin(),
                   sampled_gradients.begin() + num_samples - topK,
                   sampled_gradients.end());

  // threshold is an estimate for the kth largest element in the gradients
  // matrix
  T threshold = sampled_gradients[num_samples - topK];
  return threshold;
}

/*
 * Given the number of chunks, we split a vector into n almost equal chunks
 */
template <class T>
std::vector<std::vector<T>> splitVector(const std::vector<T>& vec, size_t n) {
  std::vector<std::vector<T>> out_vec;

  size_t length = vec.size() / n;
  size_t remain = vec.size() % n;

  size_t loop_size = std::min(n, vec.size());
  for (size_t i = 0; i < loop_size; ++i) {
    size_t begin = i * length + std::min(i, remain);
    size_t end = (i < remain) ? begin + length + 1 : begin + length;

    out_vec.emplace_back(
        std::vector<T>(vec.begin() + begin, vec.begin() + end));
  }
  while (out_vec.size() < n) {
    out_vec.push_back(std::vector<T>());
  }
  return out_vec;
}

}  // namespace thirdai::compression