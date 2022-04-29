#pragma once

#include <cereal/types/vector.hpp>
#include <cctype>
#include <cmath>

namespace thirdai::bolt {

enum class ActivationFunction { ReLU, Softmax, Linear };

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
  throw std::invalid_argument("'" + act_func_name +
                              "' is not a valid activation function");
}

constexpr float actFuncDerivative(float x, ActivationFunction act_func) {
  switch (act_func) {
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

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;

  SamplingConfig()
      : hashes_per_table(0), num_tables(0), range_pow(0), reservoir_size(0) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size) {}

  SamplingConfig(uint64_t dim, float sparsity) {
    if (sparsity < 1.0) {
      range_pow = (static_cast<uint32_t>(log2(dim)) / 3) * 3;
      hashes_per_table = range_pow / 3;
      reservoir_size = (dim * 4) / (1 << range_pow);
      num_tables = sparsity < 0.1 ? 256 : 64;
    } else {
      hashes_per_table = 0, num_tables = 0, range_pow = 0, reservoir_size = 0;
    }
  }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(hashes_per_table, num_tables, range_pow, reservoir_size);
  }
};

}  // namespace thirdai::bolt