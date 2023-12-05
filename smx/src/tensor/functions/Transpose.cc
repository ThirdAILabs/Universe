#include <Eigen/unsupported/Eigen/CXX11/Tensor>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <stdexcept>
#include <vector>

namespace thirdai::smx {

template <typename T, size_t NDim>
void eigenTranspose(const DenseTensor& in, DenseTensor& out,
                    const std::vector<size_t>& perm) {
  auto in_eigen = in.toEigen<T, NDim>();
  auto out_eigen = out.toEigen<T, NDim>();

  std::array<int, NDim> perm_arr;
  for (size_t i = 0; i < NDim; i++) {
    perm_arr[i] = perm[i];
  }
  out_eigen = in_eigen.shuffle(perm_arr);
}

template <typename T>
void generalTranspose(const DenseTensor& in, DenseTensor& out,
                      const std::vector<size_t>& perm) {
  const Shape& in_strides = in.strides();
  const Shape& out_strides = out.strides();

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();

  size_t ndim = in.ndim();

  for (size_t out_idx = 0; out_idx < out.shape().size(); out_idx++) {
    size_t in_idx = 0;
    size_t rem = out_idx;
    for (size_t i = 0; i < ndim; i++) {
      size_t index_along_dim = rem / out_strides[i];
      in_idx += index_along_dim * in_strides[perm[i]];
      rem -= index_along_dim * out_strides[i];
    }
    out_ptr[out_idx] = in_ptr[in_idx];
  }
}

template <typename T>
DenseTensorPtr transpose(const DenseTensor& in,
                         const std::vector<size_t>& perm) {
  auto out = DenseTensor::make(in.shape().permute(perm), in.dtype());

  // TODO(Nicholas): merge dims that are contiguous in perm and permute a view
  // with fewer dims.

  switch (in.ndim()) {
    case 1:
      throw std::invalid_argument(
          "Cannot transpose a tensor with 1 dimension.");
    case 2:
      eigenTranspose<T, 2>(in, *out, perm);
      return out;
    case 3:
      eigenTranspose<T, 3>(in, *out, perm);
      return out;
    case 4:
      eigenTranspose<T, 4>(in, *out, perm);
      return out;
    case 5:
      eigenTranspose<T, 5>(in, *out, perm);
      return out;
    case 6:
      eigenTranspose<T, 6>(in, *out, perm);
      return out;
    case 7:
      eigenTranspose<T, 7>(in, *out, perm);
      return out;

    default:
      generalTranspose<T>(in, *out, perm);
      return out;
  }
}

TensorPtr transpose(const TensorPtr& tensor, const std::vector<size_t>& perm) {
  if (tensor->isSparse()) {
    throw std::invalid_argument(
        "Transpose is not yet supported on sparse tensors.");
  }

  auto dense = asDense(tensor);

  switch (dense->dtype()) {
    case Dtype::f32:
      return transpose<float>(*dense, perm);
    case Dtype::u32:
      return transpose<uint32_t>(*dense, perm);
    default:
      throw std::invalid_argument("Dtype " + toString(dense->dtype()) +
                                  " is not yet supported for transpose.");
  }
}

}  // namespace thirdai::smx