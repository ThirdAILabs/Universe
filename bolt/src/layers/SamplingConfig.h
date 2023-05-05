#pragma once

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

  virtual hashing::HashFunctionPtr getHashFunction(
      uint32_t input_dim) const = 0;

  virtual hashtable::SampledHashTablePtr getHashTable() const = 0;

  virtual bool isRandomSampling() const { return false; }

  virtual ~SamplingConfig() = default;

 protected:
 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using SamplingConfigPtr = std::shared_ptr<SamplingConfig>;

class DWTASamplingConfig final : public SamplingConfig {
 public:
  DWTASamplingConfig(uint32_t num_tables, uint32_t hashes_per_table,
                     uint32_t range_pow, uint32_t binsize,
                     uint32_t reservoir_size, uint32_t permutations)
      : _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _range_pow(range_pow),
        _binsize(binsize),
        _reservoir_size(reservoir_size),
        _permutes(permutations) {}

  hashing::HashFunctionPtr getHashFunction(uint32_t input_dim) const final;

  hashtable::SampledHashTablePtr getHashTable() const final;

  static SamplingConfigPtr newAutotune(uint32_t layer_dim, float sparsity);

  static SamplingConfigPtr autotune(uint32_t layer_dim, float sparsity,
                                    bool experimental_autotune);

 private:
  uint32_t _num_tables, _hashes_per_table, _range_pow, _binsize,
      _reservoir_size, _permutes;

  // Private constructor for cereal.
  DWTASamplingConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class FastSRPSamplingConfig final : public SamplingConfig {
 public:
  FastSRPSamplingConfig(uint32_t num_tables, uint32_t hashes_per_table,
                        uint32_t reservoir_size)
      : _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _reservoir_size(reservoir_size) {}

  hashing::HashFunctionPtr getHashFunction(uint32_t input_dim) const final;

  hashtable::SampledHashTablePtr getHashTable() const final;

 private:
  uint32_t _num_tables, _hashes_per_table, _reservoir_size;

  // Private constructor for cereal.
  FastSRPSamplingConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class RandomSamplingConfig final : public SamplingConfig {
 public:
  RandomSamplingConfig() {}

  hashing::HashFunctionPtr getHashFunction(uint32_t input_dim) const final {
    (void)input_dim;
    return nullptr;
  }

  hashtable::SampledHashTablePtr getHashTable() const final { return nullptr; }

  bool isRandomSampling() const final { return true; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt
