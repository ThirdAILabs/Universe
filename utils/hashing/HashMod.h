#pragma once
#include "HashFunction.h"

namespace thirdai::utils {

template <typename LSHFunc_t>
class HashMod final : public HashFunction {
 public:
  template <typename... Args>
  HashMod(uint32_t mod, Args&&... args)
      : _mod(mod), _hash_func(std::forward<Args>(args)...) {}

  void hashSparse(uint64_t num_vectors, const uint32_t* const* indices,
                  const float* const* values, const uint32_t* lengths,
                  uint32_t* output) const override {
    _hash_func.hashSparse(num_vectors, indices, values, lengths, output);
    for (uint64_t i = 0; i < num_vectors * _hash_func.numTables(); i++) {
      output[i] = output[i] % _mod;
    }
  }

  void hashDense(uint64_t num_vectors, uint64_t dim, const float* const* values,
                 uint32_t* output) const override {
    _hash_func.hashDense(num_vectors, dim, values, output);
    for (uint64_t i = 0; i < num_vectors * _hash_func.numTables(); i++) {
      output[i] = output[i] % _mod;
    }
  }

  uint32_t numTables() const override { return _hash_func.numTables(); }

  uint32_t range() const override { return _hash_func.range(); }

 private:
  uint32_t _mod;
  LSHFunc_t _hash_func;
};

}  // namespace thirdai::utils
