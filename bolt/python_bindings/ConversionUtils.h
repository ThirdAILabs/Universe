#pragma once

#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace thirdai::bolt::python {

using thirdai::dataset::python::NumpyArray;

class BoltDatasetNumpyContext {
  /*
   * The purpose of this class is to make sure that a BoltDataset constructed
   * from a numpy array is memory safe by ensuring that the numpy arrays it is
   * constructed from cannot go out of scope while the dataset is in scope. This
   * problem arrises because if the numpy arrays passed in are not uint32 or
   * float32 then when we cast to that array type a copy will occur. This
   * resulting copy of the array will be a local copy, and thus when the method
   * constructing the dataset returns, the copy will go out of scope and the
   * dataset will be invalidated. This solves that issue.
   */
 public:
  dataset::BoltDatasetPtr dataset;

  explicit BoltDatasetNumpyContext()
      : dataset(nullptr),
        dataset_context_1(std::nullopt),
        dataset_context_2(std::nullopt) {}

  explicit BoltDatasetNumpyContext(dataset::BoltDatasetPtr&& _dataset)
      : dataset(_dataset),
        dataset_context_1(std::nullopt),
        dataset_context_2(std::nullopt) {}

  explicit BoltDatasetNumpyContext(NumpyArray<float>& examples,
                                   uint32_t batch_size)
      : dataset_context_2(std::nullopt) {
    dataset = dataset::python::denseBoltDatasetFromNumpy(examples, batch_size);
    dataset_context_1 = examples.request();
  }

  explicit BoltDatasetNumpyContext(NumpyArray<uint32_t>& labels,
                                   uint32_t batch_size)
      : dataset_context_2(std::nullopt) {
    dataset = dataset::python::categoricalLabelsFromNumpy(labels, batch_size);
    dataset_context_1 = labels.request();
  }

  explicit BoltDatasetNumpyContext(NumpyArray<uint32_t>& indices,
                                   NumpyArray<float>& values,
                                   NumpyArray<uint32_t>& offsets,
                                   uint32_t batch_size) {
    dataset = dataset::python::sparseBoltDatasetFromNumpy(indices, values,
                                                          offsets, batch_size);
    dataset_context_1 = indices.request();
    dataset_context_2 = values.request();
  }

 private:
  std::optional<py::buffer_info> dataset_context_1;
  std::optional<py::buffer_info> dataset_context_2;
};

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
    ssize_t dim, const std::string& matrix_type) {
  if (biases.ndim() != 1) {
    std::stringstream err;
    err << "Expected " << matrix_type
        << " to have 1 dimension, received matrix "
           "with "
        << biases.ndim() << " dimensions.";
    throw std::invalid_argument(err.str());
  }
  if (biases.shape(0) != dim) {
    std::stringstream err;
    err << "Expected " << matrix_type << " to have dim " << dim
        << " received matrix with dim " << biases.shape(0) << ".";
    throw std::invalid_argument(err.str());
  }
}

inline void weightDimensionCheck(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        new_weights,
    ssize_t dim, ssize_t prev_dim, const std::string& matrix_type = "") {
  if (new_weights.ndim() != 2) {
    std::stringstream err;
    err << "Expected " << matrix_type
        << " to have 2 dimensions, received matrix "
           "with "
        << new_weights.ndim() << " dimensions.";
    throw std::invalid_argument(err.str());
  }
  if (new_weights.shape(0) != dim || new_weights.shape(1) != prev_dim) {
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

inline bool isBoltDataset(const py::object& obj) {
  return py::str(obj.get_type())
      .equal(py::str("<class 'thirdai._thirdai.dataset.BoltDataset'>"));
}

inline bool isMLMDataset(const py::object& obj) {
  return py::str(obj.get_type())
      .equal(py::str("<class 'thirdai._thirdai.dataset.MLMDataset'>"));
}

inline bool isTuple(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'tuple'>"));
}

inline bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

inline py::str getDtype(const py::object& obj) {
  return py::str(obj.attr("dtype"));
}

inline bool checkNumpyDtype(const py::object& obj, const std::string& type) {
  return getDtype(obj).equal(py::str(type));
}

inline bool checkNumpyDtypeUint32(const py::object& obj) {
  return checkNumpyDtype(obj, "uint32");
}

inline bool checkNumpyDtypeFloat32(const py::object& obj) {
  return checkNumpyDtype(obj, "float32");
}

inline bool checkNumpyDtypeAnyInt(const py::object& obj) {
  return checkNumpyDtype(obj, "int32") || checkNumpyDtype(obj, "uint32") ||
         checkNumpyDtype(obj, "int64") || checkNumpyDtype(obj, "uint64");
}

inline BoltDatasetNumpyContext convertTupleToBoltDataset(const py::object& obj,
                                                         uint32_t batch_size) {
  if (batch_size == 0) {
    throw std::invalid_argument("No batch size provided.");
  }
  py::tuple tup = obj.cast<py::tuple>();
  if (tup.size() != 3) {
    throw std::invalid_argument(
        "Expected tuple of 3 numpy arrays (indices, values, offsets), "
        "received "
        "tuple of length: " +
        std::to_string(tup.size()));
  }

  if (!isNumpyArray(tup[0]) || !isNumpyArray(tup[1]) || !isNumpyArray(tup[2])) {
    throw std::invalid_argument(
        "Expected tuple of 3 numpy arrays (indices, values, offsets), "
        "received non numpy array.");
  }

  if (!checkNumpyDtypeUint32(tup[0])) {
    printCopyWarning("indices", getDtype(tup[0]), "uint32");
  }
  if (!checkNumpyDtypeFloat32(tup[1])) {
    printCopyWarning("values", getDtype(tup[1]), "float32");
  }
  if (!checkNumpyDtypeUint32(tup[2])) {
    printCopyWarning("offsets", getDtype(tup[2]), "uint32");
  }

  NumpyArray<uint32_t> indices = tup[0].cast<NumpyArray<uint32_t>>();
  NumpyArray<float> values = tup[1].cast<NumpyArray<float>>();
  NumpyArray<uint32_t> offsets = tup[2].cast<NumpyArray<uint32_t>>();

  return BoltDatasetNumpyContext(indices, values, offsets, batch_size);
}

inline BoltDatasetNumpyContext convertNumpyArrayToBoltDataset(
    const py::object& obj, uint32_t batch_size, bool is_labels) {
  if (batch_size == 0) {
    throw std::invalid_argument("No batch size provided.");
  }

  if (is_labels && checkNumpyDtypeAnyInt(obj)) {
    if (!checkNumpyDtypeUint32(obj)) {
      printCopyWarning("labels", getDtype(obj), "uint32");
    }
    auto labels = obj.cast<NumpyArray<uint32_t>>();
    return BoltDatasetNumpyContext(labels, batch_size);
  }

  if (!checkNumpyDtypeFloat32(obj)) {
    printCopyWarning("data", getDtype(obj), "float32");
  }

  NumpyArray<float> data = obj.cast<NumpyArray<float>>();
  return BoltDatasetNumpyContext(data, batch_size);
}

inline BoltDatasetNumpyContext convertPyObjectToBoltDataset(
    const py::object& obj, std::optional<uint32_t> batch_size, bool is_labels) {
  if (isBoltDataset(obj)) {
    // TODO(josh): Add a check here asserting batch size is std::nullopt once
    // we deprecate the old api
    return BoltDatasetNumpyContext(obj.cast<dataset::BoltDatasetPtr>());
  }
  if (!batch_size) {
    throw std::invalid_argument(
        "You need to set a batch size if you are passing in numpy arrays for "
        "training");
  }
  if (isNumpyArray(obj)) {
    return convertNumpyArrayToBoltDataset(obj, *batch_size, is_labels);
  }
  if (isTuple(obj)) {
    return convertTupleToBoltDataset(obj, *batch_size);
  }

  throw std::invalid_argument(
      "Expected object of type BoltDataset, tuple, or numpy array (or None "
      "for "
      "test labels), received " +
      py::str(obj.get_type()).cast<std::string>());
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

}  // namespace thirdai::bolt::python