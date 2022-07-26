#pragma once

#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/Datasets.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace thirdai::bolt::python {

// Takes in the activations arrays (if they were allocated) and returns the
// python tuple containing the metrics computed, along with the activations
// and active neurons if those are not nullptrs. Note that just the
// active_neuron pointer can be null if the output is dense. The
// activation_handle object is a python object that owns the data for
// the activations array, and likewise for the active_neuron_handle.
inline py::tuple constructPythonInferenceTuple(
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

// Helper method where the handles for active_neurons and activations are
// automatically constructed assuming no other c++ object owns their memory
inline py::tuple constructPythonInferenceTuple(py::dict&& py_metric_data,
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

}  // namespace thirdai::bolt::python