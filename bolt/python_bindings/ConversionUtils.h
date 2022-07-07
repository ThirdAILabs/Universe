#pragma once

#include <dataset/python_bindings/DatasetPython.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

inline void biasDimensionCheck(const py::array_t<float, py::array::c_style | py::array::forcecast>&
          new_biases, int64_t dim){
    if (new_biases.ndim() != 1) {
      std::stringstream err;
      err << "Expected bias matrix to have 1 dimension, received matrix "
             "with "
          << new_biases.ndim() << " dimensions.";
      throw std::invalid_argument(err.str());
    }
    if (new_biases.shape(0) != dim) {
      std::stringstream err;
      err << "Expected bias matrix to have dim " << dim
          << " received matrix with dim " << new_biases.shape(0) << ".";
      throw std::invalid_argument(err.str());
    }
  }

  inline void weightDimensionCheck(const py::array_t<float, py::array::c_style | py::array::forcecast>&
        new_weights, int64_t dim, int64_t prev_dim){
          if (new_weights.ndim() != 2) {
      std::stringstream err;
      err << "Expected weight matrix to have 2 dimensions, received matrix "
             "with "
          << new_weights.ndim() << " dimensions.";
      throw std::invalid_argument(err.str());
    }
    if (new_weights.shape(0) != dim || new_weights.shape(1) != prev_dim) {
      std::stringstream err;
      err << "Expected weight matrix to have dim (" << dim << ", " << prev_dim
          << ") received matrix with dim (" << new_weights.shape(0) << ", "
          << new_weights.shape(1) << ").";
      throw std::invalid_argument(err.str());
    }
  }

inline void biasGradientDimensionCheck(const py::array_t<float, py::array::c_style | py::array::forcecast>&
        new_biases_gradients, int64_t dim){
          if (new_biases_gradients.ndim() != 1) {
      std::stringstream err;
      err << "Expected bias gradients matrix to have 1 dimension, received "
             "matrix "
             "with "
          << new_biases_gradients.ndim() << " dimensions.";
      throw std::invalid_argument(err.str());
    }
    if (new_biases_gradients.shape(0) != dim) {
      std::stringstream err;
      err << "Expected bias gradients matrix to have dim " << dim
          << " received matrix with dim " << new_biases_gradients.shape(0)
          << ".";
      throw std::invalid_argument(err.str());
    }
        }

inline void weightGradientDimensionCheck(const py::array_t<float, py::array::c_style | py::array::forcecast>&
      new_weights_gradients, int64_t dim, int64_t prev_dim){
        if (new_weights_gradients.ndim() != 2) {
      std::stringstream err;
      err << "Expected weight gradients matrix to have 2 dimensions, received "
             "matrix "
             "with "
          << new_weights_gradients.ndim() << " dimensions.";
      throw std::invalid_argument(err.str());
    }
    if (new_weights_gradients.shape(0) != dim ||
        new_weights_gradients.shape(1) != prev_dim) {
      std::stringstream err;
      err << "Expected weight gradients matrix to have dim (" << dim << ", "
          << prev_dim << ") received matrix with dim ("
          << new_weights_gradients.shape(0) << ", "
          << new_weights_gradients.shape(1) << ").";
      throw std::invalid_argument(err.str());
    }

      }

inline bool isBoltDataset(const py::object& obj) {
  return py::str(obj.get_type())
      .equal(py::str("<class 'thirdai._thirdai.dataset.BoltDataset'>"));
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
    const py::object& obj, uint32_t batch_size, bool is_labels) {
  if (isBoltDataset(obj)) {
    return BoltDatasetNumpyContext(obj.cast<dataset::BoltDatasetPtr>());
  }
  if (isNumpyArray(obj)) {
    return convertNumpyArrayToBoltDataset(obj, batch_size, is_labels);
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