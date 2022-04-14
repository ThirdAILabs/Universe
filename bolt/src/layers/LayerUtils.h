#pragma once

#include <cereal/types/vector.hpp>

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

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(hashes_per_table, num_tables, range_pow, reservoir_size);
  }
};

}  // namespace thirdai::bolt