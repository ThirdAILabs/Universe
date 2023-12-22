#pragma once

#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::smx {

class CsrTensor final : public Tensor {
 public:
  CsrTensor(DenseTensorPtr row_offsets, DenseTensorPtr col_indices,
            const DenseTensorPtr& col_values, const Shape& dense_shape);

  static std::shared_ptr<CsrTensor> make(DenseTensorPtr row_offsets,
                                         DenseTensorPtr col_indices,
                                         const DenseTensorPtr& col_values,
                                         const Shape& dense_shape);

  static std::shared_ptr<CsrTensor> make(
      const std::vector<uint32_t>& row_offsets,
      const std::vector<uint32_t>& col_indices,
      const std::vector<float>& col_values, const Shape& dense_shape);

  const DenseTensorPtr& rowOffsets() const { return _row_offsets; }

  const DenseTensorPtr& colIndices() const { return _col_indices; }

  const DenseTensorPtr& colValues() const { return _col_values; }

  size_t nRows() const { return shape(0); }

  size_t nDenseCols() const { return shape(1); }

  bool isSparse() const final { return true; }

 private:
  DenseTensorPtr _row_offsets;
  DenseTensorPtr _col_indices;
  DenseTensorPtr _col_values;
};

using CsrTensorPtr = std::shared_ptr<CsrTensor>;

inline CsrTensorPtr csr(const TensorPtr& tensor) {
  if (auto ptr = std::dynamic_pointer_cast<CsrTensor>(tensor)) {
    return ptr;
  }
  throw std::invalid_argument("Cannot convert dense tensor to csr tensor.");
}

}  // namespace thirdai::smx