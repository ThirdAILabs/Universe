#include "DatasetPython.h"
#include <chrono>

namespace thirdai::dataset::python {

void createDatasetSubmodule(py::module_& module) {
  auto dataset_submodule = module.def_submodule("dataset");

  py::class_<InMemoryDataset<SparseBatch>> _imsd_(dataset_submodule,
                                                  "InMemorySparseDataset");
  (void)_imsd_;  // To get rid of clang tidy error

  py::class_<InMemoryDataset<DenseBatch>> _imdd_(dataset_submodule,
                                                 "InMemoryDenseDataset");
  (void)_imdd_;  // To get rid of clang tidy error

  dataset_submodule.def("load_click_through_dataset", &loadClickThroughDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("num_dense_features"),
                        py::arg("num_categorical_features"));

  py::class_<
      thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>>
      _imctd_(dataset_submodule, "ClickThroughDataset");
  (void)_imctd_;  // To get rid of clang tidy error.

  dataset_submodule.def("load_svm_dataset", &loadSVMDataset,
                        py::arg("filename"), py::arg("batch_size"));

  dataset_submodule.def("load_csv_dataset", &loadCSVDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("delimiter") = ",");
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

InMemoryDataset<DenseBatch> loadCSVDataset(const std::string& filename,
                                           uint32_t batch_size,
                                           std::string delimiter) {
  auto start = std::chrono::high_resolution_clock::now();
  InMemoryDataset<DenseBatch> data(
      filename, batch_size,
      thirdai::dataset::CsvDenseBatchFactory(delimiter.at(0)));
  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Read " << data.len() << " vectors in "
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

InMemoryDataset<DenseBatch> denseInMemoryDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size, uint64_t starting_id) {
  // Get information from examples
  const py::buffer_info examples_buf = examples.request();
  const auto examples_shape = examples_buf.shape;
  if (examples_shape.size() != 2) {
    throw std::invalid_argument(
        "For now, Numpy dense data must be 2D (each row is a dense data "
        "vector).");
  }

  uint64_t num_examples = static_cast<uint64_t>(examples_shape.at(0));
  uint64_t dimension = static_cast<uint64_t>(examples_shape.at(1));
  float* examples_raw_data = static_cast<float*>(examples_buf.ptr);

  // Get information from labels

  const py::buffer_info labels_buf = labels.request();
  const auto labels_shape = labels_buf.shape;
  if (labels_shape.size() != 1) {
    throw std::invalid_argument(
        "For now, Numpy labels must be 1D (each element is an integer).");
  }

  uint64_t num_labels = static_cast<uint64_t>(labels_shape.at(0));
  if (num_labels != num_examples) {
    throw std::invalid_argument(
        "The size of the label array must be equal to the number of rows in "
        "the examples array.");
  }
  uint32_t* labels_raw_data = static_cast<uint32_t*>(labels_buf.ptr);

  // Build batches

  uint64_t num_batches = (num_examples + batch_size - 1) / batch_size;
  std::vector<DenseBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<DenseVector> batch_vectors;
    std::vector<std::vector<uint32_t>> batch_labels;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      // owns_data = false because we don't want the numpy array to be deleted
      // if this batch (and thus the underlying vectors) get deleted
      bool owns_data = false;
      batch_vectors.emplace_back(
          dimension, examples_raw_data + dimension * vec_idx, owns_data);
      batch_labels.push_back({labels_raw_data[vec_idx]});
    }

    batches.emplace_back(std::move(batch_vectors), std::move(batch_labels),
                         starting_id + start_vec_idx);
  }

  return InMemoryDataset(std::move(batches), num_examples);
}

InMemoryDataset<SparseBatch> sparseInMemoryDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_idxs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_vals,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_offsets,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size, uint64_t starting_id) {
  // Get information from examples
  const py::buffer_info x_idxs_buf = x_idxs.request();
  const py::buffer_info x_vals_buf = x_vals.request();
  const py::buffer_info x_offsets_buf = x_offsets.request();
  const py::buffer_info labels_buf = labels.request();
  
  uint64_t num_examples = static_cast<uint64_t>(x_offsets_buf.shape.at(0)-1);
  uint32_t* x_idxs_raw_data = static_cast<uint32_t*>(x_idxs_buf.ptr);
  float* x_vals_raw_data = static_cast<float*>(x_vals_buf.ptr);
  uint32_t* x_offsets_raw_data = static_cast<uint32_t*>(x_offsets_buf.ptr);
  uint32_t* labels_raw_data = static_cast<uint32_t*>(labels_buf.ptr);

  // Get information from labels

  const auto labels_shape = labels_buf.shape;
  if (labels_shape.size() != 1) {
    throw std::invalid_argument(
        "For now, Numpy labels must be 1D (each element is an integer).");
  }

  uint64_t num_labels = static_cast<uint64_t>(labels_shape.at(0));
  if (num_labels != num_examples) {
    throw std::invalid_argument(
        "The size of the label array must be equal to the number of rows in "
        "the examples array.");
  }
  
  // Build batches

  uint64_t num_batches = (num_labels + batch_size - 1) / batch_size;
  std::vector<SparseBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<SparseVector> batch_vectors;
    std::vector<std::vector<uint32_t>> batch_labels;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      // owns_data = false because we don't want the numpy array to be deleted
      // if this batch (and thus the underlying vectors) get deleted
      bool owns_data = false;
      batch_vectors.emplace_back(
          x_idxs_raw_data+x_offsets_raw_data[vec_idx], x_vals_raw_data+x_offsets_raw_data[vec_idx], x_offsets_raw_data[vec_idx+1]-x_offsets_raw_data[vec_idx], owns_data);
      batch_labels.push_back({labels_raw_data[vec_idx]});
    }

    batches.emplace_back(std::move(batch_vectors), std::move(batch_labels),
                         starting_id + start_vec_idx);
  }

  return InMemoryDataset(std::move(batches), num_examples);
}


}  // namespace thirdai::dataset::python