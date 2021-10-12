#pragma once
#include "HashFunction.h"

namespace thirdai::utils {

template <typename LSHFunc_t>
class HashModPow2 final : public HashFunction {
 public:
  template <typename... Args>
  explicit HashModPow2(uint32_t output_bits, Args&&... args)
      : HashFunction(0, 0),
        _rshift(32 - output_bits),
        _hash_func(std::forward<Args>(args)...) {
    // We can assign these variables until the underlying hash function is
    // created, this is a hack to write to the read only parameter.
    uint32_t* num_tables_p = const_cast<uint32_t*>(&this->_num_tables);
    *num_tables_p = _hash_func.numTables();
    uint32_t* range_p = const_cast<uint32_t*>(&this->_range);
    *range_p = std::min<uint32_t>(1 << output_bits, _hash_func.range());
  }

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override {
    _hash_func.hashSingleSparse(indices, values, length, output);
    for (uint64_t i = 0; i < _hash_func.numTables(); i++) {
      output[i] = output[i] >> _rshift;
    }
  }

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override {
    _hash_func.hashSingleDense(values, dim, output);
    for (uint64_t i = 0; i < _hash_func.numTables(); i++) {
      output[i] = output[i] >> _rshift;
    }
  }

 private:
  uint32_t _rshift;
  LSHFunc_t _hash_func;
};

}  // namespace thirdai::utils
