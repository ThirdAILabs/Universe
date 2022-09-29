#pragma once

#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/Datasets.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::python {

inline void printMemoryWarning(uint64_t num_samples, uint64_t inference_dim) {
  std::cout << "Memory Error: Cannot allocate (" << num_samples << " x "
            << inference_dim
            << ") array for activations. Predict will return None for "
               "activations. Please breakup your test set into smaller pieces "
               "if you would like to have activations returned."
            << std::endl;
}

inline void printCopyWarning(const std::string& array_name,
                             const py::str& dtype_recv,
                             const std::string& dtype_expected) {
  std::cout << "Warning: " << array_name << " array has dtype=" << dtype_recv
            << " but " << dtype_expected
            << " was expected. This will result in a copy of "
               "the array in order to ensure type safety. Try specifying "
               "the dtype of the array or use .astype(...)."
            << std::endl;
}

inline void biasDimensionCheck(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& biases,
    uint64_t dim, const std::string& matrix_type) {
  if (biases.ndim() != 1) {
    std::stringstream err;
    err << "Expected " << matrix_type
        << " to have 1 dimension, received matrix "
           "with "
        << biases.ndim() << " dimensions.";
    throw std::invalid_argument(err.str());
  }
  if (biases.shape(0) != static_cast<uint32_t>(dim)) {
    std::stringstream err;
    err << "Expected " << matrix_type << " to have dim " << dim
        << " received matrix with dim " << biases.shape(0) << ".";
    throw std::invalid_argument(err.str());
  }
}

inline void weightDimensionCheck(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        new_weights,
    uint64_t dim, uint64_t prev_dim, const std::string& matrix_type = "") {
  if (new_weights.ndim() != 2) {
    std::stringstream err;
    err << "Expected " << matrix_type
        << " to have 2 dimensions, received matrix "
           "with "
        << new_weights.ndim() << " dimensions.";
    throw std::invalid_argument(err.str());
  }
  if (new_weights.shape(0) != static_cast<uint32_t>(dim) ||
      new_weights.shape(1) != static_cast<uint32_t>(prev_dim)) {
    std::stringstream err;
    err << "Expected " << matrix_type << " to have dim (" << dim << ", "
        << prev_dim << ") received matrix with dim (" << new_weights.shape(0)
        << ", " << new_weights.shape(1) << ").";
    throw std::invalid_argument(err.str());
  }
}

inline void layerIndexCheck(uint32_t layer_index, uint32_t num_layers) {
  if (layer_index >= num_layers) {
    std::stringstream err;
    err << "Expect layer_index " << num_layers << ", got " << layer_index;
    throw std::invalid_argument(err.str());
  }
}

//  Checks that the dimensions of the given numpy array match the expected
//  dimensions.
inline void checkNumpyArrayDimensions(
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

inline void allocateActivations(uint64_t num_samples, uint64_t inference_dim,
                                uint32_t** active_neurons, float** activations,
                                bool output_sparse) {
  // We use a uint64_t here in case there is overflow when we multiply the two
  // quantities. If it's larger than a uint32_t then we skip allocating since
  // this would be a 16Gb array, and could potentially mess up indexing in other
  // parts of the code.
  uint64_t total_size = num_samples * inference_dim;
  if (total_size > std::numeric_limits<uint32_t>::max()) {
    printMemoryWarning(num_samples, inference_dim);
  }
  try {
    if (output_sparse) {
      *active_neurons = new uint32_t[total_size];
    }
    *activations = new float[total_size];
  } catch (std::bad_alloc& e) {
    printMemoryWarning(num_samples, inference_dim);
  }
}

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

inline static py::array_t<float, py::array::c_style | py::array::forcecast>
denseBoltVectorToNumpy(const BoltVector& output) {
  assert(output.isDense());

  float* activations;
  allocateActivations(/* num_samples= */ 1, output.len,
                      /* active_neurons= */ nullptr, &activations,
                      /* output_sparse= */ false);

  const float* start = output.activations;
  std::copy(start, start + output.len, activations);

  py::object activation_handle = py::capsule(
      activations, [](void* ptr) { delete static_cast<float*>(ptr); });

  py::array_t<float, py::array::c_style | py::array::forcecast>
      activations_array({output.len}, {sizeof(float)}, activations,
                        activation_handle);
  return activations_array;
}

// This struct is used to wrap a char* into a stream, see
// https://stackoverflow.com/questions/7781898/get-an-istream-from-a-char
struct Membuf : std::streambuf {
  Membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

// This function defines the pickle method for a type, assuming that type
// has a static load method that takes in an istream and a save method that
// takes in an ostream.
// Pybind pickling reference:
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
// py::bytes -> char*:
// https://github.com/pybind/pybind11/issues/2517
// char* -> istream:
// https://stackoverflow.com/questions/7781898/get-an-istream-from-a-char
template <typename SERIALIZE_T>
pybind11::detail::initimpl::pickle_factory<
    std::function<py::bytes(const SERIALIZE_T&)>,
    std::function<std::shared_ptr<SERIALIZE_T>(
        py::bytes)>> inline static getPickleFunction() {
  return py::pickle<std::function<py::bytes(const SERIALIZE_T&)>,
                    std::function<std::shared_ptr<SERIALIZE_T>(py::bytes)>>(
      [](const SERIALIZE_T& model) {
        std::stringstream ss;
        model.save_stream(ss);
        std::string binary_model = ss.str();
        return py::bytes(binary_model);
      },
      [](const py::bytes& binary_model_python) {  // __setstate__
        py::buffer_info info(py::buffer(binary_model_python).request());
        char* binary_model = reinterpret_cast<char*>(info.ptr);
        Membuf sbuf(binary_model, binary_model + info.size);
        std::istream in(&sbuf);
        return SERIALIZE_T::load_stream(in);
      });
}

}  // namespace thirdai::bolt::python