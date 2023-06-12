#include "PybindUtils.h"

namespace thirdai::bolt::python {

void checkNumpyArrayDimensions(
    const std::vector<uint32_t>& expected_dimensions,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        numpy_array) {
  uint32_t numpy_array_dim = numpy_array.ndim();
  if (numpy_array_dim != expected_dimensions.size()) {
    throw std::invalid_argument(
        "Expected " + std::to_string(expected_dimensions.size()) +
        "D numpy array but received " + std::to_string(numpy_array.ndim()) +
        "D numpy array");
  }

  for (uint32_t dim_index = 0; dim_index < numpy_array_dim; dim_index++) {
    if (expected_dimensions[dim_index] != numpy_array.shape(dim_index)) {
      throw std::invalid_argument(
          "Expected dimension " + std::to_string(dim_index) + " to be " +
          std::to_string(expected_dimensions[dim_index]) +
          " but received dimension " +
          std::to_string(numpy_array.shape(dim_index)));
    }
  }
}

py::tuple constructPythonInferenceTuple(
    py::dict&& py_metric_data, uint32_t num_samples, uint32_t inference_dim,
    const float* activations, const uint32_t* active_neurons,
    const py::object& activation_handle,
    const py::object& active_neuron_handle) {
  if (!activations) {
    return py::make_tuple(py_metric_data);
  }

  // Here, we pass in the activation_handle object to the array constructor
  // for the activation array, since it is a py::object that owns the memory
  // pointed to by the raw activations pointer. This allows python to add an
  // additional reference to the activation_handle object, which will keep
  // the activation_handle py::object alive at least until the py::array goes
  // out of scope (the python garbage collection may then delete the object, but
  // not before). This results in a py::array where the memory is held by an
  // existing different python object, while still guaranteeing the memory will
  // remain valid for the lifetime of the py::array.
  py::array_t<float, py::array::c_style | py::array::forcecast>
      activations_array({num_samples, inference_dim},
                        {inference_dim * sizeof(float), sizeof(float)},
                        activations, activation_handle);

  if (!active_neurons) {
    return py::make_tuple(py_metric_data, activations_array);
  }

  // See comment above activations_array for the python memory reasons behind
  // passing in active_neuron_handle
  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>
      active_neurons_array({num_samples, inference_dim},
                           {inference_dim * sizeof(uint32_t), sizeof(uint32_t)},
                           active_neurons, active_neuron_handle);
  return py::make_tuple(py_metric_data, activations_array,
                        active_neurons_array);
}

py::tuple constructPythonInferenceTuple(py::dict&& py_metric_data,
                                        uint32_t num_samples,
                                        uint32_t inference_dim,
                                        const float* activations,
                                        const uint32_t* active_neurons) {
  // These ternary operators ensure we are not creating a python capsule of a
  // nullptr, which python doesn't like (even if we never use that capsule as a
  // handle)
  py::object activation_handle =
      (activations != nullptr)
          ? py::capsule(activations,
                        [](void* ptr) { delete static_cast<float*>(ptr); })
          : static_cast<py::object>(py::none());
  py::object active_neuron_handle =
      (active_neurons != nullptr)
          ? py::capsule(active_neurons,
                        [](void* ptr) { delete static_cast<float*>(ptr); })
          : static_cast<py::object>(py::none());
  return constructPythonInferenceTuple(
      std::move(py_metric_data), num_samples, inference_dim, activations,
      active_neurons,
      /* activation_handle = */ activation_handle,
      /* active_neuron_handle = */ active_neuron_handle);
}

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
