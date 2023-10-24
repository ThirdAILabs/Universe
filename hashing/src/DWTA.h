#pragma once

#include <cereal/types/polymorphic.hpp>
#include "HashFunction.h"
#include "proto/hashing.pb.h"
#include <utils/Random.h>
#include <vector>

namespace thirdai::hashing {

class DWTAHashFunction final : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_hashes, _dim, _binsize, _log_binsize,
      _permute;
  std::vector<uint32_t> _bin_map;
  std::vector<uint32_t> _positions;
  uint32_t _rand_double_hash_seed;

  void compactHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  DWTAHashFunction() : HashFunction(0, 0){};

 public:
  DWTAHashFunction(uint32_t input_dim, uint32_t _hashes_per_table,
                   uint32_t _num_tables, uint32_t range_pow, uint32_t binsize,
                   std::optional<uint32_t> permutations,
                   uint32_t seed = global_random::nextSeed());

  explicit DWTAHashFunction(const proto::hashing::DWTA& dwta_proto);

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  std::unique_ptr<HashFunction> copyWithNewSeeds() const final {
    return std::make_unique<DWTAHashFunction>(
        /* input_dim= */ _dim, /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables,
        /* range_pow= */ _log_binsize * _hashes_per_table,
        /* binsize=*/_binsize, /* permutations=*/
        _permute);
  }

  std::string getName() const final { return "DWTA"; }

  uint32_t getNumPermutations() const { return _permute; }

  uint32_t getBinsize() const { return _binsize; }

  uint32_t getHashesPerTable() const { return _hashes_per_table; }

  proto::hashing::HashFunction* toProto() const final;

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<DWTAHashFunction> load(const std::string& filename);

  static std::shared_ptr<DWTAHashFunction> load_stream(
      std::istream& input_stream);
};

}  // namespace thirdai::hashing
