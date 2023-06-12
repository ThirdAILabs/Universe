#pragma once

#include <cereal/types/polymorphic.hpp>
#include "HashFunction.h"
#include "MurmurHash.h"
#include <utils/Random.h>
#include <cstdint>

namespace thirdai::hashing {

/** Based off of the paper https://arxiv.org/pdf/1703.04664.pdf */
class DensifiedMinHash final : public HashFunction {
 public:
  DensifiedMinHash(uint32_t hashes_per_table, uint32_t num_tables,
                   uint32_t range, uint32_t seed = global_random::nextSeed());

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  std::unique_ptr<HashFunction> copyWithNewSeeds() const final {
    return std::make_unique<DensifiedMinHash>(
        /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables,
        /* range= */ _range);
  }

  std::string getName() const final { return "DensifiedMinhash"; }

 private:
  uint32_t _hashes_per_table, _total_num_hashes, _binsize, _seed;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HashFunction>(this), _hashes_per_table,
            _total_num_hashes, _binsize, _seed);
  }

  // constructor for cereal
  DensifiedMinHash() : HashFunction(0, 0){};
};

}  // namespace thirdai::hashing

CEREAL_REGISTER_TYPE(thirdai::hashing::DensifiedMinHash)
