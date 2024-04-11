#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>

namespace thirdai::smx {

TensorPtr concatDense(const std::vector<DenseTensorPtr>& tensors, size_t dim) {
  CHECK(!tensors.empty(), "Cannot concatenate an empty list of tensors.");
  CHECK(dim < tensors.at(0)->ndim(),
        "Cannot concat tensors with " + std::to_string(tensors.at(0)->ndim()) +
            " dimensions along dimension " + std::to_string(dim) + ".");

  size_t concat_size = 0;
  for (const auto& tensor : tensors) {
    for (size_t i = 0; i < tensors[0]->ndim(); i++) {
      if (i == dim) {
        concat_size += tensor->shape(i);
      } else {
        CHECK(tensors[0]->shape(i) == tensor->shape(i),
              "Expected tensors to have same shape except for the concat dim, "
              "found tensor with shape " +
                  tensors[0]->shape().toString() + " and tensor with shape " +
                  tensor->shape().toString() + ".");
      }
      CHECK(tensors[0]->dtype() == tensor->dtype(),
            "Expected tensors to have same dtype, found tensor with dtype " +
                toString(tensors[0]->dtype()) + " and tensor with dtype " +
                toString(tensor->dtype()) + ".");
    }
  }

  auto output = DenseTensor::make(
      tensors[0]->shape().newShapeWithDimReplaced(dim, concat_size),
      tensors[0]->dtype());

  size_t outer_size = output->shape().slice(0, dim).size();
  size_t inner_size =
      output->shape().slice(dim + 1).size() * sizeofDtype(output->dtype());
  size_t outer_stride = output->shape(dim) * inner_size;

  char* output_data = output->data<char>();

  for (size_t i = 0; i < outer_size; i++) {
    size_t output_offset = 0;
    for (const auto& tensor : tensors) {
      const char* tensor_data = tensor->data<char>();

      size_t chunk_size = tensor->shape(dim) * inner_size;

      memcpy(output_data + i * outer_stride + output_offset,
             tensor_data + i * chunk_size, chunk_size);

      output_offset += chunk_size;
    }
  }

  return output;
}

TensorPtr concat(const std::vector<TensorPtr>& tensors, size_t dim) {
  std::vector<DenseTensorPtr> dense_tensors;
  dense_tensors.reserve(tensors.size());

  for (const auto& tensor : tensors) {
    dense_tensors.push_back(dense(tensor));
  }

  return concatDense(dense_tensors, dim);
}

}  // namespace thirdai::smx