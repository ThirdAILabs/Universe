#pragma once

#include <hashing/src/MurmurHash.h>
#include <random>
#include <string>
#include <vector>

namespace thirdai::compression {

// an approximation for top-k threshold by random sampling
inline float getThresholdForTopK(const std::vector<float>& values,
                                 uint32_t sketch_size,
                                 uint32_t max_samples_for_random_sampling) {
  uint32_t num_samples = std::min(max_samples_for_random_sampling, sketch_size);
  if (num_samples < 20) {
    num_samples =
        static_cast<uint32_t>(std::min(20, static_cast<int>(values.size())));
  }
  uint32_t topK =
      static_cast<uint32_t>(1.0 * num_samples * sketch_size / values.size());

  std::vector<float> sampled_gradients(num_samples, 0);

  srand(time(0));
  for (uint32_t i = 0; i < num_samples; i++) {
    sampled_gradients[i] = std::abs(values[rand() % values.size()]);
  }

  // threshold is an estimate for the kth largest element in the gradients
  // matrix

  std::nth_element(sampled_gradients.begin(),
                   sampled_gradients.begin() + num_samples - topK,
                   sampled_gradients.end());
  float threshold = sampled_gradients[num_samples - topK];
  return threshold;
}

}  // namespace thirdai::compression