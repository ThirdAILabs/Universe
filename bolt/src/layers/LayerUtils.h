#pragma once

#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <cctype>

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
//edited
enum class hashing_function {DensifiedMinHash, DWTAHashFunction, FastSRP, MurmurHash, SparseRandomProjection, UniversalHash};

static hashing_function get_hash_function(const std::string& hash_type) {
  std::string lower_name;
  for (char c : hash_type) {
    lower_name.push_back(std::tolower(c));
  }
  if (lower_name == "densifiedminhash") {
    return hashing_function::DensifiedMinHash;
  }
  if (lower_name == "dwta") {
    return hashing_function::DWTAHashFunction;
  }
  if (lower_name == "fastsrp") {
    return hashing_function::FastSRP;
  }
  if (lower_name == "murmurhash") {
    return hashing_function::MurmurHash;
  }
  if (lower_name == "srp") {
    return hashing_function::SparseRandomProjection;
  }
  if (lower_name == "universalhash") {
    return hashing_function::UniversalHash;
  }
  throw std::invalid_argument("'" + hash_type +
                              "' is not a valid hashing function");
}
//edited
struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;
  std::string hash_type;//edited

  SamplingConfig()
      : hashes_per_table(0), num_tables(0), range_pow(0), reservoir_size(0), hash_type("DWTA") {}//edited

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size),
        hash_type("DWTA") {}//edited
  //edited
  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size, std::string hash_type_)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size),
        hash_type(std::move(hash_type_)) {}
  //edited


 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(hashes_per_table, num_tables, range_pow, reservoir_size);
  }
};

}  // namespace thirdai::bolt