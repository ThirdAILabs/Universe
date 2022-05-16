#include "DatasetPython.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <chrono>

namespace thirdai::dataset::python {

using bolt::BoltVector;

void createDatasetSubmodule(py::module_& module) {
  auto dataset_submodule = module.def_submodule("dataset");

  py::class_<BoltVector>(dataset_submodule, "BoltVector")
      .def("to_string", &BoltVector::toString)
      .def("__str__", &BoltVector::toString)
      .def("__repr__", &BoltVector::toString);

  // The no lint below is because clang tidy doesn't like the anonymous object
  // instatiation and is worried it will be lonely.
  py::class_<InMemoryDataset<SparseBatch>>(dataset_submodule,  // NOLINT
                                           "InMemorySparseDataset");

  // The no lint below is because clang tidy doesn't like the anonymous object
  // instatiation and is worried it will be lonely.
  py::class_<InMemoryDataset<DenseBatch>>(dataset_submodule,  // NOLINT
                                          "InMemoryDenseDataset");

  dataset_submodule.def("load_svm_dataset", &loadSVMDataset,
                        py::arg("filename"), py::arg("batch_size"));

  dataset_submodule.def("load_csv_dataset", &loadCSVDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("delimiter") = ",");

  dataset_submodule.def("make_sparse_vector", &BoltVector::makeSparseVector,
                        py::arg("indices"), py::arg("values"));

  dataset_submodule.def("make_dense_vector", &BoltVector::makeDenseVector,
                        py::arg("values"));

  // The no lint below is because clang tidy doesn't like the anonymous object
  // instatiation and is worried it will be lonely.
  py::class_<  // NOLINT
      thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>,
      std::shared_ptr<thirdai::dataset::InMemoryDataset<
          thirdai::dataset::ClickThroughBatch>>>(dataset_submodule,
                                                 "ClickThroughDataset");

  dataset_submodule.def(
      "load_click_through_dataset", &loadClickThroughDatasetWrapper,
      py::arg("filename"), py::arg("batch_size"),
      py::arg("num_numerical_features"), py::arg("num_categorical_features"),
      py::arg("categorical_labels"),
      "Loads a Clickthrough dataset from a file. To be used with DLRM. \n"
      "Each line of the input file should follow this format:\n"
      "```\n"
      "l\td_1\td_2\t...\td_m\tc_1\tc_2\t...\tc_n"
      "```\n"
      "where `l` is the label, `d` is a numerical (quantitative) feature, `m` "
      "is the "
      "expected number of numerical features, `c` is a categorical feature "
      "(integer only), and `n` is "
      "the expected number of categorical features.\n\n"
      "Arguments:\n"
      " * filename: String - Path to input file.\n"
      " * batch_size: Int (positive) - Size of each batch in the dataset.\n"
      " * num_numerical_features: Int (positive) - Number of expected "
      "numerical features in each dataset.\n"
      " * num_categorical_features: Int (positive) - Number of expected "
      "categorical features in each dataset.\n"
      " * categorical_labels: Boolean - True if the labels are categorical "
      "(i.e. a label of 1 means the sample "
      "belongs to category 1), False if the labels are numerical (i.e. a label "
      "of 1 means the sample corresponds "
      "with the value of 1 on the real number line).\n"
      "Each line of the input file should follow this format:\n"
      "```\n"
      ""
      "```");

  // The no lint below is because clang tidy doesn't like the anonymous object
  // instatiation and is worried it will be lonely.
  py::class_<BoltDataset, BoltDatasetPtr>(dataset_submodule,  // NOLINT
                                          "BoltDataset");

  dataset_submodule.def(
      "load_bolt_svm_dataset", &loadBoltSvmDatasetWrapper, py::arg("filename"),
      py::arg("batch_size"),
      "Loads a BoltDataset from an SVM file. Each line in the "
      "input file represents a sparse input vector and should follow this "
      "format:\n"
      "```\n"
      "l_0,l_1,...,l_m\ti_0:v_0\ti_1:v_1\t...\ti_n:v_n\n"
      "```\n"
      "where `l_0,l_1,...,l_m` is an arbitrary number of categorical "
      "labels (integers only), and each `i:v` is an index-value pair "
      "representing "
      "a non-zero element in the vector. There can be an arbitrary number "
      "of these index-value pairs.\n\n"
      "Arguments:\n"
      " * filename: String - Path to input file.\n"
      " * batch_size: Int (positive) - Size of each batch in the dataset.");

  dataset_submodule.def(
      "load_bolt_csv_dataset", &loadBoltCsvDatasetWrapper, py::arg("filename"),
      py::arg("batch_size"), py::arg("delimiter") = ",",
      "Loads a BoltDataset from a CSV file. Each line in the "
      "input file consists of a categorical label (integer) followed by the "
      "elements of the input vector (float). These numbers are separated by a "
      "delimiter."
      "Arguments:\n"
      " * filename: String - Path to input file.\n"
      " * batch_size: Int (positive) - Size of each batch in the dataset.\n"
      " * delimiter: Char - Delimiter that separates the numbers in each CSV "
      "line. Defaults to ','");

  dataset_submodule.def(
      "bolt_tokenizer", &parseSentenceToSparseArray, py::arg("sentence"),
      py::arg("seed") = 0, py::arg("dimension") = 100000,
      "Utility that turns a sentence into a sequence of token embeddings. To "
      "be used for text classification tasks.\n"
      "Arguments:\n"
      " * sentence: String - Sentence to be tokenized.\n"
      " * seed: Int - (Optional) The tokenizer uses a random number generator "
      "that needs to be seeded. Defaults to 0.\n"
      " * dimensions: Int (positive) - (Optional) The dimension of each token "
      "embedding. "
      "Defaults to 100,000.");
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

py::tuple loadBoltSvmDatasetWrapper(const std::string& filename,
                                    uint32_t batch_size) {
  auto res = loadBoltSvmDataset(filename, batch_size);
  return py::make_tuple(std::move(res.data), std::move(res.labels));
}

py::tuple loadBoltCsvDatasetWrapper(const std::string& filename,
                                    uint32_t batch_size, char delimiter) {
  auto res = loadBoltCsvDataset(filename, batch_size, delimiter);
  return py::make_tuple(std::move(res.data), std::move(res.labels));
}

py::tuple loadClickThroughDatasetWrapper(const std::string& filename,
                                         uint32_t batch_size,
                                         uint32_t num_dense_features,
                                         uint32_t num_categorical_features,
                                         bool sparse_labels) {
  auto res = loadClickThroughDataset(filename, batch_size, num_dense_features,
                                     num_categorical_features, sparse_labels);
  return py::make_tuple(std::move(res.data), std::move(res.labels));
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

BoltDatasetPtr denseBoltDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    uint32_t batch_size) {
  // Get information from examples
  const py::buffer_info examples_buf = examples.request();
  if (examples_buf.shape.size() > 2) {
    throw std::invalid_argument(
        "For now, Numpy dense data must be 1D or 2D (each row is a dense data "
        "vector).");
  }

  uint64_t num_examples = static_cast<uint64_t>(examples_buf.shape.at(0));

  // If it is a 1D array then we know the dimension is 1.
  uint64_t dimension = examples_buf.shape.size() == 2
                           ? static_cast<uint64_t>(examples_buf.shape.at(1))
                           : 1;
  float* examples_raw_data = static_cast<float*>(examples_buf.ptr);

  // Build batches

  uint64_t num_batches = (num_examples + batch_size - 1) / batch_size;
  std::vector<bolt::BoltBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_vectors;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      batch_vectors.emplace_back(
          nullptr, examples_raw_data + dimension * vec_idx, nullptr, dimension);
    }

    batches.emplace_back(std::move(batch_vectors));
  }

  return std::make_shared<BoltDataset>(std::move(batches), num_examples);
}

InMemoryDataset<SparseBatch> sparseInMemoryDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_idxs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x_vals,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_offsets,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_idxs,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_offsets,
    uint32_t batch_size, uint64_t starting_id) {
  // Get information from examples
  const py::buffer_info x_idxs_buf = x_idxs.request();
  const py::buffer_info x_vals_buf = x_vals.request();
  const py::buffer_info x_offsets_buf = x_offsets.request();
  const py::buffer_info y_idxs_buf = y_idxs.request();
  const py::buffer_info y_offsets_buf = y_offsets.request();

  uint64_t num_examples = static_cast<uint64_t>(x_offsets_buf.shape.at(0) - 1);
  uint32_t* x_idxs_raw_data = static_cast<uint32_t*>(x_idxs_buf.ptr);
  float* x_vals_raw_data = static_cast<float*>(x_vals_buf.ptr);
  uint32_t* x_offsets_raw_data = static_cast<uint32_t*>(x_offsets_buf.ptr);
  uint32_t* y_idxs_raw_data = static_cast<uint32_t*>(y_idxs_buf.ptr);
  uint32_t* y_offsets_raw_data = static_cast<uint32_t*>(y_offsets_buf.ptr);

  // Get information from labels

  uint64_t num_labels = static_cast<uint64_t>(y_offsets_buf.shape.at(0) - 1);
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
          x_idxs_raw_data + x_offsets_raw_data[vec_idx],
          x_vals_raw_data + x_offsets_raw_data[vec_idx],
          x_offsets_raw_data[vec_idx + 1] - x_offsets_raw_data[vec_idx],
          owns_data);
      std::vector<uint32_t> vec_labels;
      for (uint64_t nnz_id = y_offsets_raw_data[vec_idx];
           nnz_id < y_offsets_raw_data[vec_idx + 1]; ++nnz_id) {
        vec_labels.push_back(y_idxs_raw_data[nnz_id]);
      }
      batch_labels.push_back(std::move(vec_labels));
    }

    batches.emplace_back(std::move(batch_vectors), std::move(batch_labels),
                         starting_id + start_vec_idx);
  }

  return InMemoryDataset(std::move(batches), num_examples);
}

BoltDatasetPtr sparseBoltDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        indices,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& values,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        offsets,
    uint32_t batch_size) {
  // Get information from examples
  const py::buffer_info indices_buf = indices.request();
  const py::buffer_info values_buf = values.request();
  const py::buffer_info offsets_buf = offsets.request();

  uint64_t num_examples = static_cast<uint64_t>(offsets_buf.shape.at(0) - 1);

  uint32_t* indices_raw_data = static_cast<uint32_t*>(indices_buf.ptr);
  float* values_raw_data = static_cast<float*>(values_buf.ptr);
  uint32_t* offsets_raw_data = static_cast<uint32_t*>(offsets_buf.ptr);

  // Build batches

  uint64_t num_batches = (num_examples + batch_size - 1) / batch_size;
  std::vector<bolt::BoltBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_vectors;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      // owns_data = false because we don't want the numpy array to be deleted
      // if this batch (and thus the underlying vectors) get deleted
      auto vector_length =
          offsets_raw_data[vec_idx + 1] - offsets_raw_data[vec_idx];
      batch_vectors.emplace_back(indices_raw_data + offsets_raw_data[vec_idx],
                                 values_raw_data + offsets_raw_data[vec_idx],
                                 nullptr, vector_length);
    }

    batches.emplace_back(std::move(batch_vectors));
  }

  return std::make_shared<BoltDataset>(std::move(batches), num_examples);
}

BoltDatasetPtr categoricalLabelsFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size) {
  const py::buffer_info labels_buf = labels.request();

  if (labels_buf.shape.size() != 1) {
    throw std::invalid_argument("Expected 1D array of categorical labels.");
  }
  uint64_t num_labels = labels_buf.shape.at(0);
  uint64_t num_batches = (num_labels + batch_size - 1) / batch_size;

  const uint32_t* labels_raw_data =
      static_cast<const uint32_t*>(labels_buf.ptr);

  std::vector<bolt::BoltBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_labels;

    uint32_t end = std::min<uint32_t>(num_labels, (batch_idx + 1) * batch_size);
    for (uint32_t i = batch_idx * batch_size; i < end; i++) {
      uint32_t label = labels_raw_data[i];
      batch_labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
    }
    batches.emplace_back(std::move(batch_labels));
  }

  return std::make_shared<BoltDataset>(std::move(batches), num_labels);
}

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>>
parseSentenceToSparseArray(const std::string& sentence, uint32_t seed,
                           uint32_t dimension) {
  std::stringstream ss(sentence);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  std::vector<std::string> tokens(begin, end);

  std::unordered_map<uint32_t, uint32_t> idx_to_val_map;

  for (auto& s : tokens) {
    const char* cstr = s.c_str();
    uint32_t hash =
        thirdai::hashing::MurmurHash(cstr, s.length(), seed) % dimension;
    if (idx_to_val_map.find(hash) == idx_to_val_map.end()) {
      idx_to_val_map[hash] = 1;
    } else {
      idx_to_val_map[hash]++;
    }
  }

  auto result = py::array_t<uint32_t>(idx_to_val_map.size());
  py::buffer_info indx_buf = result.request();
  uint32_t* indx_ptr = static_cast<uint32_t*>(indx_buf.ptr);

  auto result_2 = py::array_t<uint32_t>(idx_to_val_map.size());
  py::buffer_info val_buf = result_2.request();
  uint32_t* val_ptr = static_cast<uint32_t*>(val_buf.ptr);

  int i = 0;
  for (auto kv : idx_to_val_map) {
    indx_ptr[i] = kv.first;
    val_ptr[i] = kv.second;
    i += 1;
  }

  return std::make_tuple(result, result_2);
}

}  // namespace thirdai::dataset::python