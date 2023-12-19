#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>
#include <bolt/src/neuron_index/RandomSampler.h>
#include <hashing/src/DWTA.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

namespace thirdai::bolt {

class SamplingConfig {
 public:
  SamplingConfig() {}

  virtual NeuronIndexPtr getNeuronIndex(uint32_t layer_dim,
                                        uint32_t input_dim) const = 0;

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
  DWTASamplingConfig() {}
  DWTASamplingConfig(uint32_t num_tables, uint32_t hashes_per_table,
                     uint32_t range_pow, uint32_t binsize,
                     uint32_t reservoir_size,
                     std::optional<uint32_t> permutations)
      : _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _range_pow(range_pow),
        _binsize(binsize),
        _reservoir_size(reservoir_size),
        _permutes(permutations) {}

  NeuronIndexPtr getNeuronIndex(uint32_t layer_dim,
                                uint32_t input_dim) const final;

  uint32_t reservoirSize() const { return _reservoir_size; }

  hashing::HashFunctionPtr getHashFunction(uint32_t input_dim) const;

  hashtable::SampledHashTablePtr getHashTable() const;

  static std::shared_ptr<DWTASamplingConfig> newAutotune(uint32_t layer_dim,
                                                         float sparsity);

  static std::shared_ptr<DWTASamplingConfig> oldAutotune(uint32_t layer_dim,
                                                         float sparsity);

  static std::shared_ptr<DWTASamplingConfig> autotune(
      uint32_t layer_dim, float sparsity, bool experimental_autotune);

 private:
  uint32_t _num_tables, _hashes_per_table, _range_pow, _binsize,
      _reservoir_size;
  std::optional<uint32_t> _permutes;

  // Private constructor for cereal.

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

  NeuronIndexPtr getNeuronIndex(uint32_t layer_dim,
                                uint32_t input_dim) const final;

  hashing::HashFunctionPtr getHashFunction(uint32_t input_dim) const;

  hashtable::SampledHashTablePtr getHashTable() const;

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

  NeuronIndexPtr getNeuronIndex(uint32_t layer_dim,
                                uint32_t input_dim) const final {
    (void)input_dim;
    return RandomSampler::make(layer_dim);
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt
