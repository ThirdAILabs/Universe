#include "CsrTensor.h"

namespace thirdai::smx {

CsrTensor::CsrTensor(DenseTensorPtr row_offsets, DenseTensorPtr col_indices,
                     const DenseTensorPtr& col_values, const Shape& dense_shape)
    : Tensor(dense_shape, col_values->dtype()),
      _row_offsets(std::move(row_offsets)),
      _col_indices(std::move(col_indices)),
      _col_values(col_values) {
  CHECK(_shape.ndim() == 2, "Csr Tensors must be 2d.");

  CHECK(_row_offsets->dtype() == Dtype::u32, "Row Offsets must be type u32.");
  CHECK(_col_indices->dtype() == Dtype::u32, "Col Indices must be type u32.");

  CHECK(_row_offsets->ndim() == 1, "Row Offsets Tensors must be 1d.");
  CHECK(_col_indices->ndim() == 1, "Col Indices Tensors must be 1d.");
  CHECK(_col_values->ndim() == 1, "Col Values Tensors must be 1d.");

  CHECK(_row_offsets->shape(0) == shape(0) + 1,
        "Row offsets must have size shape[0] + 1.");
  CHECK(_col_indices->shape(0) == _col_values->shape(0),
        "Num indices and values must match.");

  size_t indices_len = _col_indices->size();
  size_t indices_dim = _shape.last();
  const uint32_t* indices_data = _col_indices->data<uint32_t>();
  for (size_t i = 0; i < indices_len; i++) {
    CHECK(indices_data[i] < indices_dim,
          "Invalid index " + std::to_string(indices_data[i]) +
              " for CsrTensor with dim " + std::to_string(indices_dim) + ".");
  }

  size_t offsets_len = _row_offsets->size();
  const uint32_t* offsets_data = _row_offsets->data<uint32_t>();
  size_t last_offset = offsets_data[0];
  CHECK(last_offset == 0, "First offset in CsrTensor should be 0.");
  for (size_t i = 1; i < offsets_len; i++) {
    CHECK(last_offset <= offsets_data[i],
          "offsets[i+1] should be >= offsets[i] in CsrTensor.");
    last_offset = offsets_data[i];
  }
  CHECK(last_offset == indices_len,
        "Last offset in CsrTensor should match the end of the indices/values.");
}

std::shared_ptr<CsrTensor> CsrTensor::make(DenseTensorPtr row_offsets,
                                           DenseTensorPtr col_indices,
                                           const DenseTensorPtr& col_values,
                                           const Shape& dense_shape) {
  return std::make_shared<CsrTensor>(
      std::move(row_offsets), std::move(col_indices), col_values, dense_shape);
}

std::shared_ptr<CsrTensor> CsrTensor::make(
    const std::vector<uint32_t>& row_offsets,
    const std::vector<uint32_t>& col_indices,
    const std::vector<float>& col_values, const Shape& dense_shape) {
  return make(DenseTensor::make(row_offsets, Shape(row_offsets.size())),
              DenseTensor::make(col_indices, Shape(col_indices.size())),
              DenseTensor::make(col_values, Shape(col_values.size())),
              dense_shape);
}

}  // namespace thirdai::smx