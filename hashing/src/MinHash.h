#pragma once

#include <cereal/types/polymorphic.hpp>
#include "HashFunction.h"
#include "UniversalHash.h"
#include <proto/hashing.pb.h>
#include <utils/Random.h>

namespace thirdai::hashing {

class MinHash final : public HashFunction {
 public:
  MinHash(uint32_t hashes_per_table, uint32_t num_tables, uint32_t range,
          uint32_t seed = global_random::nextSeed());

  explicit MinHash(const proto::hashing::MinHash& minhash);

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  std::unique_ptr<HashFunction> copyWithNewSeeds() const final {
    return std::make_unique<MinHash>(
        /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables,
        /* range= */ _range);
  }

  proto::hashing::HashFunction* toProto() const final;

  std::string getName() const final { return "Minhash"; }

 private:
  uint32_t _hashes_per_table;
  uint32_t _total_num_hashes;
  std::vector<UniversalHash> _hash_functions;

  MinHash() : HashFunction(0, 0){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HashFunction>(this), _hashes_per_table,
            _total_num_hashes, _hash_functions);
  }
};

}  // namespace thirdai::hashing

CEREAL_REGISTER_TYPE(thirdai::hashing::MinHash)