#pragma once

#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

enum class SparsityType {
  Sparse,
  Dense,
  Unknown,
};

class InputTensor final : public Tensor {
 public:
  InputTensor(uint32_t dim, SparsityType sparsity_type,
              std::optional<uint32_t> num_nonzeros);

  std::optional<uint32_t> numNonzeros() const final;

  BoltVector& getVector(uint32_t index) final;

  void setInputs(BoltBatch& batch);

  SparsityType sparsityType() const;

 private:
  BoltBatch* _input_batch;

  std::optional<uint32_t> _num_nonzeros;
  SparsityType _sparsity_type;
};

using InputTensorPtr = std::shared_ptr<InputTensor>;

}  // namespace thirdai::bolt::nn::tensor