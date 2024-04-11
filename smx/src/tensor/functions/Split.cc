#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>
#include <numeric>

namespace thirdai::smx {

std::vector<TensorPtr> splitDense(const DenseTensorPtr& tensor, size_t dim,
                                  const std::vector<size_t>& sizes) {
  CHECK(dim < tensor->ndim(),
        "Cannot split tensor with " + std::to_string(tensor->ndim()) +
            " dimensions along dimension " + std::to_string(dim));

  CHECK(std::reduce(sizes.begin(), sizes.end()) == tensor->shape(dim),
        "Sum of split sizes must match shape of tensor along the split "
        "dimension.");

  size_t outer_size = tensor->shape().slice(0, dim).size();
  size_t inner_size =
      tensor->shape().slice(dim + 1).size() * sizeofDtype(tensor->dtype());
  size_t outer_stride = tensor->shape(dim) * inner_size;

  const char* tensor_data = tensor->data<char>();

  std::vector<TensorPtr> outputs;

  size_t offset = 0;
  for (size_t size : sizes) {
    CHECK(size > 0, "Cannot have 0 size in tensor split.");

    auto output = DenseTensor::make(
        tensor->shape().newShapeWithDimReplaced(dim, size), tensor->dtype());

    char* output_data = output->data<char>();

    size_t chunk_size = size * inner_size;
    for (size_t i = 0; i < outer_size; i++) {
      memcpy(output_data + i * chunk_size,
             tensor_data + i * outer_stride + offset, chunk_size);
    }

    offset += chunk_size;

    outputs.push_back(output);
  }

  return outputs;
}

std::vector<TensorPtr> split(const TensorPtr& tensor, size_t dim,
                             const std::vector<size_t>& sizes) {
  return splitDense(dense(tensor), dim, sizes);
}

}  // namespace thirdai::smx