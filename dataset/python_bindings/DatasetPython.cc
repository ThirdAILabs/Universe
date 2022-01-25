#include "DatasetPython.h"
#include <chrono>

namespace thirdai::dataset::python {

void createDatasetSubmodule(py::module_& module) {
  auto dataset_submodule = module.def_submodule("dataset");

  py::class_<InMemoryDataset<SparseBatch>> _imsd_(dataset_submodule,
                                                  "InMemorySparseDataset");
  (void)_imsd_;  // To get rid of clang tidy error

  dataset_submodule.def("loadClickThroughDataset", &loadClickThroughDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("num_dense_features"),
                        py::arg("num_categorical_features"));

  py::class_<
      thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>>
      _imctd_(dataset_submodule, "ClickThroughDataset");
  (void)_imctd_;  // To get rid of clang tidy error.

  dataset_submodule.def("loadSVMDataset", &loadSVMDataset, py::arg("filename"),
                        py::arg("batch_size"));
}

InMemoryDataset<ClickThroughBatch> loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features) {
  auto start = std::chrono::high_resolution_clock::now();
  thirdai::dataset::ClickThroughBatchFactory factory(num_dense_features,
                                                     num_categorical_features);
  InMemoryDataset<ClickThroughBatch> data(filename, batch_size,
                                          std::move(factory));
  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Read " << data.len() << " vectors from " << filename << " in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  return data;
}

InMemoryDataset<SparseBatch> loadSVMDataset(const std::string& filename,
                                            uint32_t batch_size) {
  auto start = std::chrono::high_resolution_clock::now();
  InMemoryDataset<SparseBatch> data(filename, batch_size,
                                    thirdai::dataset::SvmSparseBatchFactory{});
  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Read " << data.len() << " vectors from " << filename << " in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;

  return data;
}

// TODO(josh): Is this method in a good place?
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
// for explanation of why we do py::array::c_style and py::array::forcecase.
// Ensures array is an array of floats in dense row major order.
SparseBatch wrapNumpyIntoSparseData(
    const std::vector<py::array_t<
        float, py::array::c_style | py::array::forcecast>>& sparse_values,
    const std::vector<
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
        sparse_indices,
    uint64_t starting_id) {
  if (sparse_values.size() != sparse_indices.size()) {
    throw std::invalid_argument(
        "Values and indices arrays must have the same number of elements.");
  }

  uint64_t num_vectors = sparse_values.size();

  std::vector<dataset::SparseVector> batch_vectors;
  for (uint64_t vec_id = 0; vec_id < num_vectors; vec_id++) {
    const py::buffer_info indices_buf = sparse_indices.at(vec_id).request();
    const py::buffer_info values_buf = sparse_values.at(vec_id).request();
    const auto indices_shape = indices_buf.shape;
    const auto values_shape = values_buf.shape;

    if (indices_shape.size() != 1 || values_shape.size() != 1) {
      throw std::invalid_argument(
          "For now, every entry in the indices and values arrays must be a 1D "
          "array.");
    }

    if (indices_shape.at(0) != values_shape.at(0)) {
      throw std::invalid_argument(
          "Corresponding indice and value entries must have the same number of "
          "values.");
    }

    bool owns_data = false;
    uint64_t length = indices_shape.at(0);
    batch_vectors.emplace_back(static_cast<uint32_t*>(indices_buf.ptr),
                               static_cast<float*>(values_buf.ptr), length,
                               owns_data);
  }

  return SparseBatch(std::move(batch_vectors), starting_id);
}

DenseBatch wrapNumpyIntoDenseBatch(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& data,
    uint64_t starting_id) {
  const py::buffer_info data_buf = data.request();
  const auto shape = data_buf.shape;
  if (shape.size() != 2) {
    throw std::invalid_argument(
        "For now, Numpy dense data must be 2D (each row is a dense data "
        "vector).");
  }

  uint64_t num_vectors = static_cast<uint64_t>(shape.at(0));
  uint64_t dimension = static_cast<uint64_t>(shape.at(1));
  float* raw_data = static_cast<float*>(data_buf.ptr);

  std::vector<dataset::DenseVector> batch_vectors;
  for (uint64_t vec_id = 0; vec_id < num_vectors; vec_id++) {
    // owns_data = false because we don't want the numpy array to be deleted
    // if this batch (and thus the underlying vectors) get deleted
    bool owns_data = false;
    batch_vectors.emplace_back(dimension, raw_data + dimension * vec_id,
                               owns_data);
  }

  return DenseBatch(std::move(batch_vectors), starting_id);
}

}  // namespace thirdai::dataset::python