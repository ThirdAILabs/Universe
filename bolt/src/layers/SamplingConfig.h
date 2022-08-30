#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class SamplingConfig {
 public:
  SamplingConfig() {}

  virtual std::unique_ptr<hashing::HashFunction> getHashFunction(
      uint32_t input_dim) const = 0;

  virtual std::unique_ptr<hashtable::SampledHashTable<uint32_t>> getHashTable()
      const = 0;

  virtual bool isRandomSampling() const { return false; }

 protected:
  static uint32_t clip(uint32_t input, uint32_t low, uint32_t high) {
    if (input < low) {
      return low;
    }
    if (input > high) {
      return high;
    }
    return input;
  }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using SamplingConfigPtr = std::shared_ptr<SamplingConfig>;

class DWTASamplingConfig final : public SamplingConfig {
 public:
  DWTASamplingConfig(uint32_t num_tables, uint32_t hashes_per_table,
                     uint32_t reservoir_size)
      : _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _reservoir_size(reservoir_size) {}

  std::unique_ptr<hashing::HashFunction> getHashFunction(
      uint32_t input_dim) const final {
    return std::make_unique<hashing::DWTAHashFunction>(
        /* input_dim= */ input_dim,
        /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables,
        /* range_pow= */ 3 * _num_tables);
  }

  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> getHashTable()
      const final {
    return std::make_unique<hashtable::SampledHashTable<uint32_t>>(
        /* num_tables= */ _num_tables,
        /* reservoir_size= */ _reservoir_size,
        /* range= */ 1 << (3 * _hashes_per_table));
  }

  static SamplingConfigPtr autotune(uint32_t layer_dim, float sparsity) {
    if (sparsity == 1.0) {
      // If the layer is dense then we don't need to create a sampling config
      // for it.
      return nullptr;
    }

    // The number of items in the table is equal to the number of neurons in
    // this layer, which is stored in the "dim" variable. By analyzing the
    // hash table, we find that
    // E(num_elements_per_bucket) = dim / 2^(range_pow) = sparsity * dim *
    // safety_factor / num_tables The first expression comes from analyzing a
    // single hash table, while the second comes from analyzing the total number
    // of elements returned across the tables. safety_factor is a constant that
    // equals how many more times elements we want to expect to have across
    // tables than the minimum. Simplifying, we have 1 / 2^(range_pow) =
    // sparsity * safety_factor / num_tables This leaves us with 3 free
    // variables: safety_factor, num_tables, and hashes_per_table.

    // First, we will set num_tables_guess = 128 and safety_factor = 1.
    // num_tables_guess is an initial guess to get a good value for range_pow,
    // but we do not find the final num_tables until below because the rounding
    // in the range_pow calculation step can mess things up.
    uint32_t num_tables_guess = 128;
    float safety_factor = 1;

    // We can now set range_pow: manipulating the equation, we have that
    // range_pow = log2(num_tables / (sparsity * safety_factor))
    float range_pow_float =
        std::log2(num_tables_guess / (sparsity * safety_factor));
    // By the properties of DWTA, hashes_per_table = range_pow / 3.
    float hashes_per_table_float = range_pow_float / 3;
    // We now round hashes_per_table to the nearest integer.
    // Using round is more accurate than truncating it down.
    uint32_t hashes_per_table = std::round(hashes_per_table_float);
    // Finally, hashes_per_table needs to be clipped, and then we can
    // recalculate range_pow
    hashes_per_table = clip(hashes_per_table, /* low = */ 2, /* high = */ 8);
    uint32_t range_pow = hashes_per_table * 3;

    // We now calculate an exact value for num_tables using the formula
    // num_tables = sparsity * safety_factor * 2^(range_pow)
    uint32_t num_tables =
        std::round(sparsity * safety_factor * (1 << range_pow));

    // Finally, we want to set reservoir_size to be somewhat larger than
    // the number of expected elements per bucket. Here, we choose as a
    // heuristic 4 times the number of expected elements per bucket. We take
    // a max with 1 to ensure that the reservoir size isn't 0.
    uint32_t expected_num_elements_per_bucket =
        std::max<uint32_t>(layer_dim / (1 << range_pow), 1);
    uint32_t reservoir_size = 4 * expected_num_elements_per_bucket;

    return std::make_shared<DWTASamplingConfig>(
        /* num_tables= */ num_tables,
        /* hashes_per_table= */ hashes_per_table,
        /* reservoir_size= */ reservoir_size);
  }

 private:
  uint32_t _num_tables, _hashes_per_table, _reservoir_size;

  // Private constructor for cereal.
  DWTASamplingConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<SamplingConfig>(this), _num_tables,
            _hashes_per_table, _reservoir_size);
  }
};

class FastSRPSamplingConfig final : public SamplingConfig {
 public:
  FastSRPSamplingConfig(uint32_t num_tables, uint32_t hashes_per_table,
                        uint32_t reservoir_size)
      : _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _reservoir_size(reservoir_size) {}

  std::unique_ptr<hashing::HashFunction> getHashFunction(
      uint32_t input_dim) const final {
    return std::make_unique<hashing::FastSRP>(
        /* input_dim= */ input_dim, /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables);
  }

  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> getHashTable()
      const final {
    return std::make_unique<hashtable::SampledHashTable<uint32_t>>(
        /* num_tables= */ _num_tables,
        /* reservoir_size= */ _reservoir_size,
        /* range= */ 1 << _hashes_per_table);
  }

 private:
  uint32_t _num_tables, _hashes_per_table, _reservoir_size;

  // Private constructor for cereal.
  FastSRPSamplingConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<SamplingConfig>(this), _num_tables,
            _hashes_per_table, _reservoir_size);
  }
};

class RandomSamplingConfig final : public SamplingConfig {
 public:
  RandomSamplingConfig() {}

  std::unique_ptr<hashing::HashFunction> getHashFunction(
      uint32_t input_dim) const final {
    (void)input_dim;
    return nullptr;
  }

  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> getHashTable()
      const final {
    return nullptr;
  }

  bool isRandomSampling() const final { return true; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<SamplingConfig>(this));
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DWTASamplingConfig)
CEREAL_REGISTER_TYPE(thirdai::bolt::FastSRPSamplingConfig)
CEREAL_REGISTER_TYPE(thirdai::bolt::RandomSamplingConfig)