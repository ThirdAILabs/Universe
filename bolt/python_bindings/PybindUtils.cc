#include "PybindUtils.h"

namespace thirdai::bolt::python {

NumpyArray<float> getGradients(const nn::model::ModelPtr& model) {
  auto [grads, flattened_dim] = model->getFlattenedGradients();

  py::capsule free_when_done(
      grads, [](void* ptr) { delete static_cast<float*>(ptr); });

  return NumpyArray<float>(flattened_dim, grads, free_when_done);
}

NumpyArray<float> getParameters(const nn::model::ModelPtr& model) {
  auto [grads, flattened_dim] = model->getFlattenedParameters();

  py::capsule free_when_done(
      grads, [](void* ptr) { delete static_cast<float*>(ptr); });

  return NumpyArray<float>(flattened_dim, grads, free_when_done);
}

void setGradients(const nn::model::ModelPtr& model,
                  NumpyArray<float>& new_values) {
  if (new_values.ndim() != 1) {
    throw std::invalid_argument("Expected grads to be flattened.");
  }

  uint64_t flattened_dim = new_values.shape(0);
  model->setFlattenedGradients(new_values.data(), flattened_dim);
}

void setParameters(const nn::model::ModelPtr& model,
                   NumpyArray<float>& new_values) {
  if (new_values.ndim() != 1) {
    throw std::invalid_argument("Expected params to be flattened.");
  }

  uint64_t flattened_dim = new_values.shape(0);
  model->setFlattenedParameters(new_values.data(), flattened_dim);
}

}  // namespace thirdai::bolt::python
