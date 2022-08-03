#pragma once

#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/MurmurHash.h>
#include <hashing/src/SRP.h>
#include <sys/types.h>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::bolt {

enum class ActivationFunction { ReLU, Softmax, Linear, Tanh, Sigmoid };

static std::string activationFunctionToStr(ActivationFunction act_func) {
  switch (act_func) {
    case ActivationFunction::ReLU:
      return "ReLU";
    case ActivationFunction::Softmax:
      return "Softmax";
    case ActivationFunction::Sigmoid:
      return "Sigmoid";
    case ActivationFunction::Linear:
      return "Linear";
    case ActivationFunction::Tanh:
      return "Tanh";
  }
  throw std::logic_error(
      "Invalid activation function passed to activationFunctionToStr.");
}

// an approximation for top-k threshold by random sampling
inline float getThresholdForTopK(const std::vector<float>& values,
                                 uint32_t sketch_size,
                                 uint32_t max_samples_for_random_sampling) {
  uint32_t num_samples = std::min(max_samples_for_random_sampling, sketch_size);
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

inline void getDragonSketch(const std::vector<float>& full_gradient,
                            uint32_t* indices, float* gradients,
                            int seed_for_hashing, float threshold,
                            uint32_t sketch_size,
                            bool unbiased_sketch = false) {
  uint32_t loop_size = full_gradient.size();

  if (!unbiased_sketch) {
#pragma omp parallel for default(none)                                \
    shared(indices, gradients, full_gradient, sketch_size, threshold, \
           loop_size, seed_for_hashing)
    for (uint32_t i = 0; i < loop_size; i++) {
      if (std::abs(full_gradient[i]) > threshold) {
        int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                                std::to_string(i).length(),
                                                seed_for_hashing) %
                   sketch_size;
        indices[hash] = i;
        gradients[hash] = full_gradient[i];
      }
    }
  }
}

static ActivationFunction getActivationFunction(
    const std::string& act_func_name) {
  std::string lower_name;
  for (char c : act_func_name) {
    lower_name.push_back(std::tolower(c));
  }
  if (lower_name == "relu") {
    return ActivationFunction::ReLU;
  }
  if (lower_name == "softmax") {
    return ActivationFunction::Softmax;
  }
  if (lower_name == "sigmoid") {
    return ActivationFunction::Sigmoid;
  }
  if (lower_name == "linear") {
    return ActivationFunction::Linear;
  }
  if (lower_name == "tanh") {
    return ActivationFunction::Tanh;
  }
  throw std::invalid_argument(
      "'" + act_func_name +
      "' is not a valid activation function. Supported activation functions: "
      "'relu', 'softmax', 'sigmoid', 'linear', and 'tanh'.");
}

constexpr float actFuncDerivative(float activation,
                                  ActivationFunction act_func) {
  switch (act_func) {
    case ActivationFunction::Tanh:
      // Derivative of tanh(x) is 1 - tanh^2(x), but in this case
      // activation =  tanh(x), so derivative is simply: 1 - (activation)^2.
      return (1 - activation * activation);
    case ActivationFunction::ReLU:
      return activation > 0 ? 1.0 : 0.0;
    case ActivationFunction::Sigmoid:
    // The derivative of sigmoid is computed as part of the BinaryCrossEntropy
    // loss function since they are used together and this simplifies the
    // computations.
    case ActivationFunction::Softmax:
      // The derivative of softmax is computed as part of the
      // CategoricalCrossEntropy loss function since they are used together, and
      // this simplifies the computations.
    case ActivationFunction::Linear:
      return 1.0;
  }
  // This is impossible to reach, but the compiler gave a warning saying it
  // reached the end of a non void function without it.
  return 0.0;
}

}  // namespace thirdai::bolt