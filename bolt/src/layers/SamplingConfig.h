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

  virtual ~SamplingConfig() = default;

 protected:
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
      uint32_t input_dim) const final;

  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> getHashTable()
      const final;

  static SamplingConfigPtr autotune(uint32_t layer_dim, float sparsity);

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
      uint32_t input_dim) const final;

  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> getHashTable()
      const final;

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
