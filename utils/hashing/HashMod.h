#pragma once
#include "HashFunction.h"

namespace thirdai::utils {

template <typename LSHFunc_t>
class HashMod final : public HashFunction {
 public:
  /**
   * We are using std::forward and a variadic argument construct to ensure
   * memory safety. Because we don't know if the underlying hash function
   * allocates memory and has its copy/move constructors implmented it is not
   * save to take in the hash function object directly and simply assign it to
   * _hash_func. We could take in a pointer but that means that every call the
   * hash function could require an additional memory access and also fragments
   * the memory more.
   */
  template <typename... Args>
  explicit HashMod(uint32_t mod, Args&&... args)
      : HashFunction(0, 0), _mod(mod), _hash_func(std::forward<Args>(args)...) {
    // We can assign these variables until the underlying hash function is
    // created, this is a hack to write to the read only parameter.
    uint32_t* num_tables_p = const_cast<uint32_t*>(&this->_num_tables);
    *num_tables_p = _hash_func.numTables();
    uint32_t* range_p = const_cast<uint32_t*>(&this->_range);
    *range_p = std::min<uint32_t>(mod, _hash_func.range());
  }

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override {
    _hash_func.hashSingleSparse(indices, values, length, output);
    for (uint64_t i = 0; i < _hash_func.numTables(); i++) {
      output[i] = output[i] % _mod;
    }
  }

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override {
    _hash_func.hashSingleDense(values, dim, output);
    for (uint64_t i = 0; i < _hash_func.numTables(); i++) {
      output[i] = output[i] % _mod;
    }
  }

 private:
  const uint32_t _mod;
  LSHFunc_t _hash_func;
};

}  // namespace thirdai::utils
