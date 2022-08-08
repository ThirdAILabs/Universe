#pragma once

#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/MurmurHash.h>
#include <hashing/src/SRP.h>
#include <sys/types.h>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <random>
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
  
  // std::cout<<"value of num_samples: "<<num_samples<<" value of topK: "<<topK<<std::endl;

  std::vector<float> sampled_gradients(num_samples, 0);

  srand(time(0));
  for (uint32_t i = 0; i < num_samples; i++) {
    sampled_gradients[i] = std::abs(values[rand() % values.size()]);
  }

  // threshold is an estimate for the kth largest element in the gradients
  // matrix

  // std::cout<<"printing the sampled gradients"<<std::endl;
  std::nth_element(sampled_gradients.begin(),
                   sampled_gradients.begin() + num_samples - topK,
                   sampled_gradients.end());
  // for(int i=0;i<std::min(static_cast<int>(sampled_gradients.size()),20);i++){
  //   std::cout<<sampled_gradients[sampled_gradients.size()-i-1]<<" ";
  // }
  // std::cout<<std::endl;
  float threshold = sampled_gradients[num_samples - topK];
  // std::cout<<"the value of threshold is: "<<threshold<<std::endl;

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

inline void getUnbiasedSketch(const std::vector<float>& full_gradient,
                              int* indices, int sketch_size,
                              int seed_for_hashing,
                              bool pregenerate_distribution, float threshold) {
  int loop_size = static_cast<int>(full_gradient.size());

  (void)threshold;
  (void)sketch_size;
  (void)indices;
  (void)seed_for_hashing;
  (void)full_gradient;

  std::vector<int> random_numbers;
  int range_pregenerated_numbers;

  if (pregenerate_distribution) {
    // std::cout << "Pregenerate distribution is true" << std::endl;
    int pregenerated_numbers = 10'000;
    range_pregenerated_numbers = 2048;
    std::mt19937 gen(time(0));
    std::uniform_int_distribution<> distrib(0, range_pregenerated_numbers);

    random_numbers.assign(pregenerated_numbers, 0);

#pragma omp parallel for default(none) \
    shared(random_numbers, pregenerated_numbers, distrib, gen)
    for (int i = 0; i < pregenerated_numbers; i++) {
      random_numbers[i] = distrib(gen);
    }
  }

  // std::cout << "Pregenerated numbers are done " << std::endl;

  std::mt19937 index(rand() % 1000);
  int reset_index_after = 1000;
  int current_index_reps = 1000;
  int current_index = 0;

#pragma omp parallel for default(none) \
  shared(full_gradient, indices, sketch_size, seed_for_hashing, \
  pregenerate_distribution, threshold, loop_size, random_numbers,
  \
  range_pregenerated_numbers, index) \
  firstprivate(reset_index_after, current_index_reps, current_index)

  for (int i = 0; i < loop_size; i++) {
    if (std::abs(full_gradient[i]) > threshold) {
      int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                              std::to_string(i).length(),
                                              seed_for_hashing) %
                 sketch_size;
      indices[hash] = i * ((full_gradient[i] > 0) - (full_gradient[i] < 0));
    } else {
      if (pregenerate_distribution) {
        if (current_index_reps >= reset_index_after) {
          current_index = index() % random_numbers.size();
          current_index_reps = 0;
        }
        if (random_numbers[current_index] <=
            static_cast<int>(range_pregenerated_numbers *
                             std::abs(full_gradient[i]) / threshold)) {
          int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                                  std::to_string(i).length(),
                                                  seed_for_hashing) %
                     sketch_size;
          indices[hash] = i * ((full_gradient[i] > 0) - (full_gradient[i] < 0));
          current_index = (current_index + 1) % random_numbers.size();
          current_index_reps++;
        }
      } else {
        std::bernoulli_distribution coinflip(std::abs(full_gradient[i]) /
                                             threshold);
        if (coinflip(index)) {
          int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                                  std::to_string(i).length(),
                                                  seed_for_hashing) %
                     sketch_size;
          indices[hash] = i * ((full_gradient[i] > 0) - (full_gradient[i] < 0));
        }
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