#pragma once

#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/SRP.h>
#include <cctype>
#include <string>
#include <utility>

namespace thirdai::bolt {

enum class ActivationFunction { ReLU, Softmax, Linear, Tanh, Sigmoid };

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
  throw std::invalid_argument("'" + act_func_name +
                              "' is not a valid activation function");
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

// Didn't include DensifiedMinhash because its hashSingleDense() method has not
// been implemented.
enum class HashFunctionEnum { DWTA, FastSRP, SRP };

static HashFunctionEnum getHashFunction(const std::string& hash_function) {
  std::string lower_name;
  for (char c : hash_function) {
    lower_name.push_back(std::tolower(c));
  }
  if (lower_name == "dwta") {
    return HashFunctionEnum::DWTA;
  }
  if (lower_name == "fastsrp") {
    return HashFunctionEnum::FastSRP;
  }
  if (lower_name == "srp") {
    return HashFunctionEnum::SRP;
  }
  throw std::invalid_argument(
      "'" + hash_function +
      "' is not a Supported LSH function. Supported Functions are "
      "SRP, FastSRP, DWTA");
}

inline std::string getHashString(HashFunctionEnum hash_function) {
  switch (hash_function) {
    case HashFunctionEnum::DWTA:
      return "DWTA";
    case HashFunctionEnum::SRP:
      return "SRP";
    case HashFunctionEnum::FastSRP:
      return "FastSRP";
    // Not supposed to reach here but compiler complains
    default:
      throw std::invalid_argument("Hash function not supported.");
  }
}

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;
  HashFunctionEnum _hash_function;

  SamplingConfig()
      : hashes_per_table(0),
        num_tables(0),
        range_pow(0),
        reservoir_size(0),
        _hash_function(HashFunctionEnum::DWTA) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size,
                 const std::string& hash_function = "DWTA")
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size),
        _hash_function(getHashFunction(hash_function)) {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(hashes_per_table, num_tables, range_pow, reservoir_size,
            _hash_function);
  }
};

}  // namespace thirdai::bolt