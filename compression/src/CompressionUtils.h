#pragma once

#include <hashing/src/MurmurHash.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace thirdai::compression {

// an approximation for top-k threshold by random sampling
template <class T>
inline T getThresholdForTopK(const std::vector<T>& values, uint32_t sketch_size,
                             uint32_t max_samples_for_random_sampling) {
  uint32_t num_samples = std::min(max_samples_for_random_sampling, sketch_size);
  if (num_samples < 20) {
    num_samples =
        static_cast<uint32_t>(std::min(20, static_cast<int>(values.size())));
  }
  uint32_t topK =
      static_cast<uint32_t>(1.0 * num_samples * sketch_size / values.size());

  std::vector<T> sampled_gradients(num_samples, 0);

  srand(time(0));
  for (uint32_t i = 0; i < num_samples; i++) {
    sampled_gradients[i] = std::abs(values[rand() % values.size()]);
  }

  // threshold is an estimate for the kth largest element in the gradients
  // matrix

  std::nth_element(sampled_gradients.begin(),
                   sampled_gradients.begin() + num_samples - topK,
                   sampled_gradients.end());
  T threshold = sampled_gradients[num_samples - topK];
  return threshold;
}

template <class T>
inline T getThresholdForTopK(const T* values, uint32_t size,
                             uint32_t sketch_size,
                             uint32_t max_samples_for_random_sampling) {
  uint32_t num_samples = std::min(max_samples_for_random_sampling, sketch_size);
  if (num_samples < 20) {
    num_samples = static_cast<uint32_t>(std::min(20, static_cast<int>(size)));
  }
  uint32_t topK = static_cast<uint32_t>(1.0 * num_samples * sketch_size / size);

  std::vector<T> sampled_gradients(num_samples, 0);

  srand(time(0));
  for (uint32_t i = 0; i < num_samples; i++) {
    sampled_gradients[i] = std::abs(values[rand() % size]);
  }

  // threshold is an estimate for the kth largest element in the gradients
  // matrix

  std::nth_element(sampled_gradients.begin(),
                   sampled_gradients.begin() + num_samples - topK,
                   sampled_gradients.end());
  T threshold = sampled_gradients[num_samples - topK];
  return threshold;
}

template <class T>
std::vector<std::vector<T>> SplitVector(const std::vector<T>& vec, size_t n) {
  std::vector<std::vector<T>> outVec;

  size_t length = vec.size() / n;
  size_t remain = vec.size() % n;

  size_t begin = 0;
  size_t end = 0;

  for (size_t i = 0; i < std::min(size_t(n), vec.size()); ++i) {
    end += (remain > 0) ? (length + !!(remain--)) : length;

    outVec.emplace_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

    begin = end;
  }
  while (outVec.size() < n) {
    outVec.push_back(std::vector<T>());
  }
  return outVec;
}

}  // namespace thirdai::compression