#include "DatasetPython.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_types/BoltInputBatch.h>
#include <dataset/src/utils/BuilderVectors.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/core/BatchProcessor.h>
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <dataset/src/encodings/text/TextEncodingInterface.h>
#include <dataset/src/encodings/text/CharKGram.h>
#include <dataset/src/encodings/text/BoltTokenizer.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <dataset/src/encodings/categorical/OneHotEncoding.h>
#include <pybind11/attr.h>
#include <chrono>
#include <memory>

namespace thirdai::dataset::python {

void createDatasetSubmodule(py::module_& module) {
  auto dataset_submodule = module.def_submodule("dataset");

  py::class_<BoltVector>(dataset_submodule, "BoltVector")
      .def("to_string", &BoltVector::toString)
      .def("__str__", &BoltVector::toString)
      .def("__repr__", &BoltVector::toString);

  py::class_<BoltInputBatch>(dataset_submodule, "BoltInputBatch")
      .def(py::init<std::vector<BoltVector>&&, std::vector<BoltVector>&&>(),
           py::arg("vectors"), py::arg("labels"))
      .def("to_string", &BoltInputBatch::toString)
      .def("__str__", &BoltInputBatch::toString)
      .def("__repr__", &BoltInputBatch::toString)
      .def("size", &BoltInputBatch::getBatchSize);
  
  py::class_<BuilderVector, std::shared_ptr<BuilderVector>> _builder_vector_(dataset_submodule, "__BuilderVector__",
        "An interface for builder vectors.");

  py::class_<SparseBuilderVector, BuilderVector, std::shared_ptr<SparseBuilderVector>>(dataset_submodule, "__SparseBuilderVector__")
      .def(py::init<>())
      .def("addSingleFeature", &SparseBuilderVector::addSingleFeature,
           py::arg("start_dim"), py::arg("value"))
      .def("addSparseFeatures", &SparseBuilderVector::addSparseFeaturesNumpy,
           py::arg("indices"), py::arg("values"))
      .def("addDenseFeatures", &SparseBuilderVector::addDenseFeaturesNumpy,
           py::arg("start_dim"), py::arg("values"))
      .def("to_bolt_vector", &SparseBuilderVector::toBoltVector);
      
  py::class_<DenseBuilderVector, BuilderVector, std::shared_ptr<DenseBuilderVector>>(dataset_submodule, "__DenseBuilderVector__")
      .def(py::init<>())
      .def("addSingleFeature", &DenseBuilderVector::addSingleFeature,
           py::arg("start_dim"), py::arg("value"))
      .def("addSparseFeatures", &DenseBuilderVector::addSparseFeaturesNumpy,
           py::arg("indices"), py::arg("values"))
      .def("addDenseFeatures", &DenseBuilderVector::addDenseFeaturesNumpy,
           py::arg("start_dim"), py::arg("values"))
      .def("to_bolt_vector", &DenseBuilderVector::toBoltVector);
  
  py::class_<TextEncoding, std::shared_ptr<TextEncoding>>(dataset_submodule, "TextEncoding")
      .def("embed_text", &TextEncoding::embedText, 
           py::arg("text"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("is_dense", &TextEncoding::isDense)
      .def("feature_dim", &TextEncoding::featureDim);

  py::class_<CharKGram, TextEncoding, std::shared_ptr<CharKGram>>(dataset_submodule, "CharKGram")
      .def(py::init<uint32_t, uint32_t>(), py::arg("k"), py::arg("dim"))
      .def("embed_text", &CharKGram::embedText, 
           py::arg("text"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("is_dense", &CharKGram::isDense)
      .def("feature_dim", &CharKGram::featureDim);
  
  py::class_<BoltTokenizer, TextEncoding, std::shared_ptr<BoltTokenizer>>(dataset_submodule, "BoltTokenizer")
      .def(py::init<uint32_t>(), py::arg("dim")=100000)
      .def("embed_text", &BoltTokenizer::embedText, 
           py::arg("text"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("is_dense", &BoltTokenizer::isDense)
      .def("feature_dim", &BoltTokenizer::featureDim);
  
  py::class_<PairGram, TextEncoding, std::shared_ptr<PairGram>>(dataset_submodule, "PairGram")
      .def(py::init<uint32_t>(), py::arg("dim")=100000)
      .def("embed_text", &PairGram::embedText, 
           py::arg("text"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("is_dense", &PairGram::isDense)
      .def("feature_dim", &PairGram::featureDim);

  py::class_<CategoricalEncoding, std::shared_ptr<CategoricalEncoding>>(dataset_submodule, "CategoricalEncoding")
      .def("embedCategory", &CategoricalEncoding::embedCategory, py::arg("id"), py::arg("shared_feature_vector"), py::arg("offset"))
      .def("featureDim", &CategoricalEncoding::featureDim)
      .def("isDense", &CategoricalEncoding::isDense);
  
  py::class_<OneHotEncoding, CategoricalEncoding, std::shared_ptr<OneHotEncoding>>(dataset_submodule, "OneHot")
      .def(py::init<uint32_t>(), py::arg("dim"))
      .def("embedCategory", &OneHotEncoding::embedCategory, py::arg("id"), py::arg("shared_feature_vector"), py::arg("offset"))
      .def("featureDim", &OneHotEncoding::featureDim)
      .def("isDense", &OneHotEncoding::isDense);

  py::class_<Block, std::shared_ptr<Block>>(dataset_submodule, "Block")
      .def("process", &Block::process, py::arg("input_row"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("featureDim", &Block::featureDim)
      .def("isDense", &Block::isDense);

  py::class_<TextBlock, Block, std::shared_ptr<TextBlock>>(dataset_submodule, "Text")
      .def(py::init<uint32_t, std::shared_ptr<TextEncoding>&>(), py::arg("col"), 
           py::arg("encoding"))
      .def(py::init<uint32_t, uint32_t>(), py::arg("col"), py::arg("dim"))
      .def("process", &TextBlock::process, py::arg("input_row"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("featureDim", &TextBlock::featureDim)
      .def("isDense", &TextBlock::isDense);
  
  py::class_<CategoricalBlock, Block, std::shared_ptr<CategoricalBlock>>(dataset_submodule, "Categorical")
      .def(py::init<uint32_t, std::shared_ptr<CategoricalEncoding>&, bool>(), py::arg("col"), py::arg("encoding"), py::arg("from_string")=false)
      .def(py::init<uint32_t, uint32_t, bool>(), py::arg("col"), py::arg("dim"), py::arg("from_string")=false)
      .def("process", &CategoricalBlock::process, py::arg("input_row"), py::arg("shared_feature_vector"), py::arg("idx_offset"))
      .def("featureDim", &CategoricalBlock::featureDim)
      .def("isDense", &CategoricalBlock::isDense);

  py::class_<BatchProcessor>(dataset_submodule, "BatchProcessor")
      .def(py::init<std::vector<std::shared_ptr<Block>>&, std::vector<std::shared_ptr<Block>>&, uint32_t>(),
           py::arg("input_blocks"), py::arg("target_blocks"), py::arg("output_batch_size"))
      .def("process_batch", &BatchProcessor::processBatch, py::arg("row_batch"))
      .def("export_in_memory_dataset", &BatchProcessor::exportInMemoryDataset, py::arg("shuffle")=false, py::arg("shuffle_seed")=0);

  py::class_<InMemoryDataset<SparseBatch>> _imsd_(dataset_submodule,
                                                  "InMemorySparseDataset");
  (void)_imsd_;  // To get rid of clang tidy error

  py::class_<InMemoryDataset<DenseBatch>> _imdd_(dataset_submodule,
                                                 "InMemoryDenseDataset");
  (void)_imdd_;  // To get rid of clang tidy error

  dataset_submodule.def(
      "load_click_through_dataset", &loadClickThroughDataset,
      py::arg("filename"), py::arg("batch_size"), py::arg("num_dense_features"),
      py::arg("num_categorical_features"), py::arg("sparse_labels"));

  py::class_<
      thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>>
      _imctd_(dataset_submodule, "ClickThroughDataset");
  (void)_imctd_;  // To get rid of clang tidy error.

  dataset_submodule.def("load_svm_dataset", &loadSVMDataset,
                        py::arg("filename"), py::arg("batch_size"));

  dataset_submodule.def("load_csv_dataset", &loadCSVDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("delimiter") = ",");

  dataset_submodule.def("make_sparse_vector", &BoltVector::makeSparseVector,
                        py::arg("indices"), py::arg("values"));

  dataset_submodule.def("make_dense_vector", &BoltVector::makeDenseVector,
                        py::arg("values"));

  py::class_<InMemoryDataset<BoltInputBatch>>(dataset_submodule, "BoltDataset")
      .def(py::init<std::vector<BoltInputBatch>&&, uint64_t>(),
           py::arg("batches"), py::arg("num_elements"))
      .def("at", &InMemoryDataset<BoltInputBatch>::at, py::arg("idx"));

  dataset_submodule.def("load_bolt_svm_dataset", &loadBoltSVMDataset,
                        py::arg("filename"), py::arg("batch_size"));

  dataset_submodule.def("load_bolt_csv_dataset", &loadBoltCSVDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("delimiter") = ",");

  dataset_submodule.def("bolt_tokenizer", &parseSentenceToSparseArray,
                        py::arg("sentence"), py::arg("seed") = 42,
                        py::arg("dimension") = 100000);

}

InMemoryDataset<ClickThroughBatch> loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features,
    bool sparse_labels) {
  auto start = std::chrono::high_resolution_clock::now();
  thirdai::dataset::ClickThroughBatchFactory factory(
      num_dense_features, num_categorical_features, sparse_labels);
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

InMemoryDataset<BoltInputBatch> loadBoltSVMDataset(const std::string& filename,
                                                   uint32_t batch_size) {
  auto start = std::chrono::high_resolution_clock::now();
  InMemoryDataset<BoltInputBatch> data(filename, batch_size,
                                       thirdai::dataset::BoltSvmBatchFactory{});
  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Read " << data.len() << " vectors from " << filename << " in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;

  return data;
}

InMemoryDataset<BoltInputBatch> loadBoltCSVDataset(const std::string& filename,
                                                   uint32_t batch_size,
                                                   std::string delimiter) {
  auto start = std::chrono::high_resolution_clock::now();
  InMemoryDataset<BoltInputBatch> data(
      filename, batch_size,
      thirdai::dataset::BoltCsvBatchFactory(delimiter.at(0)));
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

InMemoryDataset<BoltInputBatch> denseBoltDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size) {
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
  std::vector<BoltInputBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_vectors;
    std::vector<BoltVector> batch_labels;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      batch_vectors.emplace_back(
          nullptr, examples_raw_data + dimension * vec_idx, nullptr, dimension);
      batch_labels.push_back(
          BoltVector::makeSparseVector({labels_raw_data[vec_idx]}, {1.0}));
    }

    batches.emplace_back(std::move(batch_vectors), std::move(batch_labels));
  }

  return InMemoryDataset(std::move(batches), num_examples);
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

InMemoryDataset<BoltInputBatch> sparseBoltDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_idxs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x_vals,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_offsets,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_idxs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& y_vals,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_offsets,
    uint32_t batch_size) {
  // Get information from examples
  const py::buffer_info x_idxs_buf = x_idxs.request();
  const py::buffer_info x_vals_buf = x_vals.request();
  const py::buffer_info x_offsets_buf = x_offsets.request();
  const py::buffer_info y_idxs_buf = y_idxs.request();
  const py::buffer_info y_vals_buf = y_vals.request();
  const py::buffer_info y_offsets_buf = y_offsets.request();

  uint64_t num_examples = static_cast<uint64_t>(x_offsets_buf.shape.at(0) - 1);
  uint32_t* x_idxs_raw_data = static_cast<uint32_t*>(x_idxs_buf.ptr);
  float* x_vals_raw_data = static_cast<float*>(x_vals_buf.ptr);
  uint32_t* x_offsets_raw_data = static_cast<uint32_t*>(x_offsets_buf.ptr);
  uint32_t* y_idxs_raw_data = static_cast<uint32_t*>(y_idxs_buf.ptr);
  float* y_vals_raw_data = static_cast<float*>(y_vals_buf.ptr);
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
  std::vector<BoltInputBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_vectors;
    std::vector<BoltVector> batch_labels;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      // owns_data = false because we don't want the numpy array to be deleted
      // if this batch (and thus the underlying vectors) get deleted
      auto vector_length =
          x_offsets_raw_data[vec_idx + 1] - x_offsets_raw_data[vec_idx];
      batch_vectors.emplace_back(x_idxs_raw_data + x_offsets_raw_data[vec_idx],
                                 x_vals_raw_data + x_offsets_raw_data[vec_idx],
                                 nullptr, vector_length);
      auto label_length =
          y_offsets_raw_data[vec_idx + 1] - y_offsets_raw_data[vec_idx];
      batch_labels.emplace_back(y_idxs_raw_data + y_offsets_raw_data[vec_idx],
                                y_vals_raw_data + y_offsets_raw_data[vec_idx],
                                nullptr, label_length);
    }

    batches.emplace_back(std::move(batch_vectors), std::move(batch_labels));
  }

  return InMemoryDataset(std::move(batches), num_examples);
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