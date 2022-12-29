#pragma once

#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

enum class SparsityType {
  Sparse,
  Dense,
  Unknown,
};

/**
 * Subclass of Tensor which represents inputs to the computation graph.
 */
class InputTensor final : public Tensor {
 public:
  InputTensor(uint32_t dim, SparsityType sparsity_type,
              std::optional<uint32_t> num_nonzeros);

  static std::shared_ptr<InputTensor> make(
      uint32_t dim, SparsityType sparsity_type,
      std::optional<uint32_t> num_nonzeros);

  std::optional<uint32_t> numNonzeros() const final;

  BoltVector& getVector(uint32_t index) final;

  /**
   * Sets the batch which whose vectors will be returned by subsequent calls to
   * getVector.
   */
  void setInputs(const BoltBatch& batch);

  /**
   * Returns if the input contains vectors that are sparse, dense, or unknown
   * (meaning it is unknown if they will be sparse or dense, or they could be
   * either). Some ops like concatenation will need to know the sparsity of
   * their inputs.
   */
  SparsityType sparsityType() const;

 private:
  BoltBatch* _input_batch;

  // The number of nonzeros is optional because this may not be fixed for some
  // inputs.
  std::optional<uint32_t> _num_nonzeros;
  SparsityType _sparsity_type;
};

using InputTensorPtr = std::shared_ptr<InputTensor>;

}  // namespace thirdai::bolt::nn::tensor