#pragma once

#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/SRP.h>
#include <utils/text/StringManipulation.h>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

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

static ActivationFunction getActivationFunction(
    const std::string& act_func_name) {
  std::string lower_name = text::lower(act_func_name);
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

static void checkSparsity(float sparsity) {
  if (sparsity > 1 || sparsity <= 0) {
    throw std::invalid_argument(
        "sparsity must be between 0 exclusive and 1 inclusive.");
  }
  if (0.2 < sparsity && sparsity < 1.0) {
    std::cout << "WARNING: Using large sparsity value " << sparsity
              << " in Layer, consider decreasing sparsity" << std::endl;
  }
}

}  // namespace thirdai::bolt