#include <dataset/python_bindings/DatasetPython.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

void createBoltSubmodule(py::module_& module);

// Returns true on success and false on allocation failure.
bool allocateActivations(uint64_t num_samples, uint64_t inference_dim,
                         uint32_t** active_neurons, float** activations,
                         bool output_sparse);

// Takes in the activations arrays (if they were allocated) and returns the
// correct python tuple containing the activations (and active neurons if
// sparse) and the metrics computed.
py::tuple constructNumpyArrays(py::dict&& py_metric_data, uint32_t num_samples,
                               uint32_t inference_dim, uint32_t* active_neurons,
                               float* activations, bool output_sparse,
                               bool alloc_success);

static inline bool isBoltDataset(const py::object& obj) {
  return py::str(obj.get_type())
      .equal(py::str("<class 'thirdai._thirdai.dataset.BoltDataset'>"));
}

static inline bool isTuple(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'tuple'>"));
}

static inline bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

static inline py::str getDtype(const py::object& obj) {
  return py::str(obj.attr("dtype"));
}

static inline bool checkNumpyDtype(const py::object& obj,
                                   const std::string& type) {
  return getDtype(obj).equal(py::str(type));
}

static inline bool checkNumpyDtypeUint32(const py::object& obj) {
  return checkNumpyDtype(obj, "uint32");
}

static inline bool checkNumpyDtypeFloat32(const py::object& obj) {
  return checkNumpyDtype(obj, "float32");
}

static inline bool checkNumpyDtypeAnyInt(const py::object& obj) {
  return checkNumpyDtype(obj, "int32") || checkNumpyDtype(obj, "uint32") ||
         checkNumpyDtype(obj, "int64") || checkNumpyDtype(obj, "uint64");
}

static inline void printCopyWarning(const std::string& array_name,
                                    const py::str& dtype_recv,
                                    const std::string& dtype_expected) {
  std::cout << "Warning: " << array_name << " array has dtype=" << dtype_recv
            << " but " << dtype_expected
            << " was expected. This will result in a copy of "
               "the array in order to ensure type safety. Try specifying "
               "the dtype of the array or use .astype(...)."
            << std::endl;
}

static inline BoltDatasetNumpyContext convertTupleToBoltDataset(
    const py::object& obj, uint32_t batch_size) {
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

static inline BoltDatasetNumpyContext convertNumpyArrayToBoltDataset(
    const py::object& obj, uint32_t batch_size, bool is_labels,
    uint32_t network_input_dim) {
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
  uint32_t input_dim = data.ndim() == 1 ? 1 : data.shape(1);
  if (input_dim != network_input_dim) {
    throw std::invalid_argument(
        "Cannot pass array with input dimension " + std::to_string(input_dim) +
        " to network with input dim " + std::to_string(network_input_dim));
  }

  return BoltDatasetNumpyContext(data, batch_size);
}

static inline BoltDatasetNumpyContext convertPyObjectToBoltDataset(
    const py::object& obj, uint32_t batch_size, bool is_labels,
    uint32_t network_input_dim) {
  if (isBoltDataset(obj)) {
    return BoltDatasetNumpyContext(obj.cast<dataset::BoltDatasetPtr>());
  }
  if (isNumpyArray(obj)) {
    return convertNumpyArrayToBoltDataset(obj, batch_size, is_labels,
                                          network_input_dim);
  }
  if (isTuple(obj)) {
    return convertTupleToBoltDataset(obj, batch_size);
  }

  throw std::invalid_argument(
      "Expected object of type BoltDataset, tuple, or numpy array (or None "
      "for "
      "test labels), received " +
      py::str(obj.get_type()).cast<std::string>());
}
}  // namespace thirdai::bolt::python