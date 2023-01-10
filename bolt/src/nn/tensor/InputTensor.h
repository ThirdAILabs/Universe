#pragma once

#include "Tensor.h"
#include <optional>

namespace thirdai::bolt::nn::tensor {

/**
 * Subclass of Tensor which represents inputs to the model.
 */
class InputTensor final : public Tensor {
 public:
  /**
   * The argument num_nonzeros sets how many nonzero elements sparse inputs must
   * have. This is required for concatenation of tokens in embedding layers and
   * for concatenation ops to be able to concatenate sparse inputs. If not
   * provided the input can still accept sparse inputs, but will not preform any
   * checks on their number of nonzeros and will return std::nullopt if
   * numNonzeros() is called on it.
   */
  InputTensor(uint32_t dim, std::optional<uint32_t> num_nonzeros);

  static std::shared_ptr<InputTensor> make(
      uint32_t dim, std::optional<uint32_t> num_nonzeros = std::nullopt);

  std::optional<uint32_t> numNonzeros(bool use_sparsity) const final;

  BoltVector& getVector(uint32_t index) final;

  /**
   * Sets the batch which whose vectors will be returned by subsequent calls to
   * getVector.
   */
  void setInputs(const BoltBatch& batch);

 private:
  BoltBatch* _input_batch;

  // The number of nonzeros is optional because this may not be fixed for some
  // inputs.
  std::optional<uint32_t> _num_nonzeros;
};

using InputTensorPtr = std::shared_ptr<InputTensor>;

}  // namespace thirdai::bolt::nn::tensor