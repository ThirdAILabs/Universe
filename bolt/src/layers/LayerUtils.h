#pragma once

#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/SRP.h>
#include <cctype>
#include <utility>

namespace thirdai::bolt {

enum class ActivationFunction { ReLU, Softmax, Linear, Tanh };

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
  if (lower_name == "linear") {
    return ActivationFunction::Linear;
  }
  if (lower_name == "tanh") {
    return ActivationFunction::Tanh;
  }
  throw std::invalid_argument("'" + act_func_name +
                              "' is not a valid activation function");
}

constexpr float actFuncDerivative(float x, ActivationFunction act_func) {
  switch (act_func) {
    case ActivationFunction::Tanh:
      // Derivative of tanh(x) is 1 - tanh^2(x), but in this case x = tanh(x),
      // so derivative is simply: 1 - x^2.
      return (1 - x * x);
    case ActivationFunction::ReLU:
      return x > 0 ? 1.0 : 0.0;
    case ActivationFunction::Softmax:
      // return 1.0; // Commented out because Clang tidy doesn't like
      // consecutive identical branches
    case ActivationFunction::Linear:
      return 1.0;
      // default:
      //   return 0.0;
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

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;
  HashFunctionEnum hash_function;

  SamplingConfig()
      : hashes_per_table(0),
        num_tables(0),
        range_pow(0),
        reservoir_size(0),
        hash_function(HashFunctionEnum::DWTA) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size,
                 HashFunctionEnum hash_function = HashFunctionEnum::DWTA)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size),
        hash_function(hash_function) {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(hashes_per_table, num_tables, range_pow, reservoir_size,
            hash_function);
  }
};

}  // namespace thirdai::bolt