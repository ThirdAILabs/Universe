#include "DatasetPython.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/StreamingGenericDatasetLoader.h>
#include <dataset/src/bolt_datasets/batch_processors/MaskedSentenceBatchProcessor.h>
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/ContiguousNumericId.h>
#include <dataset/src/encodings/text/CharKGram.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <dataset/src/encodings/text/TextEncodingInterface.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
#include <dataset/src/encodings/text/UniGram.h>
#include <dataset/tests/MockBlock.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <sys/types.h>
#include <chrono>
#include <limits>
#include <type_traits>
#include <unordered_map>

// TODO(Geordie): Split into smaller files.
// I'm thinking one for each submodule of dataset_submodule.
// E.g. in DatasetBlockPython.cc we would have a function with this signature:
// void createBlockSubsubmodule(py::module_& dataset_submodule,
//                              py::module_& internal_dataset_submodule);

namespace thirdai::dataset::python {

using bolt::BoltVector;

void createDatasetSubmodule(py::module_& module) {
  // Separate submodule for bindings that we don't want to expose to users.
  auto internal_dataset_submodule = module.def_submodule("dataset_internal");

  // Everything in this submodule is exposed to users.
  auto dataset_submodule = module.def_submodule("dataset");
  auto block_submodule = dataset_submodule.def_submodule("blocks");
  auto text_encoding_submodule =
      dataset_submodule.def_submodule("text_encodings");
  auto categorical_encoding_submodule =
      dataset_submodule.def_submodule("categorical_encodings");

  py::class_<BoltVector>(dataset_submodule, "BoltVector")
      .def("to_string", &BoltVector::toString)
      .def("__str__", &BoltVector::toString)
      .def("__repr__", &BoltVector::toString);

  // The no lint below is because clang tidy doesn't like instantiating an
  // object without a name and never using it.
  py::class_<InMemoryDataset<SparseBatch>>(dataset_submodule,  // NOLINT
                                           "InMemorySparseDataset");

  // The no lint below is because clang tidy doesn't like instantiating an
  // object without a name and never using it.
  py::class_<InMemoryDataset<DenseBatch>>(dataset_submodule,  // NOLINT
                                          "InMemoryDenseDataset");

  py::class_<TextEncoding, std::shared_ptr<TextEncoding>>(
      internal_dataset_submodule, "TextEncoding",
      "Interface for text encoders.")
      .def("is_dense", &TextEncoding::isDense,
           "True if the encoder produces dense features, False otherwise.")
      .def("feature_dim", &TextEncoding::featureDim,
           "The dimension of the encoding.");

  py::class_<PairGram, TextEncoding, std::shared_ptr<PairGram>>(
      text_encoding_submodule, "PairGram",
      "Encodes a sentence as a weighted set of ordered pairs of "
      "whitespace-delimited words. Self-pairs are included. "
      "Expects a textual string, e.g. A good good model, which is then "
      "encoded as 'A A': 1, 'A good': 2, 'A model': 1, 'good good': 3, 'good "
      "model': 2, 'model model': 1.")
      .def(py::init<uint32_t>(),
           py::arg("dim") = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
           "Constructor. Accepts the desired dimension of the encoding.")
      .def("is_dense", &PairGram::isDense,
           "Returns False since this is a sparse encoding.")
      .def("feature_dim", &PairGram::featureDim,
           "The dimension of the encoding.");

  py::class_<UniGram, TextEncoding, std::shared_ptr<UniGram>>(
      text_encoding_submodule, "UniGram",
      "Encodes a sentence as a weighted set of whitespace-delimited words. "
      "Expects a textual string, e.g. A good good model, which is then "
      "encoded as 'A': 1, 'good': 2, 'model': 1.")
      .def(py::init<uint32_t>(),
           py::arg("dim") = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
           "Constructor. Accepts the desired dimension of the encoding.")
      .def("is_dense", &UniGram::isDense,
           "Returns False since this is a sparse encoding.")
      .def("feature_dim", &UniGram::featureDim,
           "The dimension of the encoding.");

  py::class_<CharKGram, TextEncoding, std::shared_ptr<CharKGram>>(
      text_encoding_submodule, "CharKGram",
      "Encodes a sentence as a weighted set of character trigrams.")
      .def(py::init<uint32_t, uint32_t>(), py::arg("k"),
           py::arg("dim") = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
           "Constructor. Accepts k (the number of characters in each token) "
           "and dimension of the encoding.")
      .def("is_dense", &CharKGram::isDense,
           "Returns False since this is a sparse encoding.")
      .def("feature_dim", &CharKGram::featureDim,
           "The dimension of the encoding.");

  py::class_<CategoricalEncoding, std::shared_ptr<CategoricalEncoding>>(
      internal_dataset_submodule, "CategoricalEncoding",
      "Interface for categorical feature encoders.")
      .def("feature_dim", &CategoricalEncoding::featureDim,
           "True if the encoder produces dense features, False otherwise.")
      .def("is_dense", &CategoricalEncoding::isDense,
           "The dimension of the encoding.");

  py::class_<ContiguousNumericId, CategoricalEncoding,
             std::shared_ptr<ContiguousNumericId>>(
      categorical_encoding_submodule, "ContiguousNumericId",
      "Expects a number and treats it as an ID in a contiguous set of "
      "numeric IDs in a given range (0-indexed, excludes end of range). "
      "If the ID is beyond the given range, it performs a modulo operation. "
      "To illustrate, if dim = 10, then 0 through 9 map to themselves, "
      "and any number n >= 10 maps to n % 10.")
      .def(py::init<uint32_t>(), py::arg("dim"),
           "Constructor. Accepts the desired dimension of the encoding.")
      .def("feature_dim", &ContiguousNumericId::featureDim,
           "Returns False since this is a sparse encoding.")
      .def("is_dense", &ContiguousNumericId::isDense,
           "The dimension of the encoding.");

  py::class_<Block, std::shared_ptr<Block>>(
      internal_dataset_submodule, "Block",
      "Block abstract class.\n\n"
      "A block accepts an input sample in the form of a sequence of strings "
      "then encodes this sequence as a vector.")
      .def("feature_dim", &Block::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &Block::isDense,
           "True if the block produces dense features, False otherwise.");

  py::class_<TextBlock, Block, std::shared_ptr<TextBlock>>(
      block_submodule, "Text",
      "A block that encodes text (e.g. sentences / paragraphs).")
      .def(py::init<uint32_t, std::shared_ptr<TextEncoding>>(), py::arg("col"),
           py::arg("encoding"),
           "Constructor.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the text to be encoded.\n"
           " * encoding: TextEncoding - Text encoding model.")
      .def(py::init<uint32_t, uint32_t>(), py::arg("col"), py::arg("dim"),
           "Constructor with default encoder.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the text to be encoded.\n"
           " * dim: Int - Dimension of the encoding")
      .def("feature_dim", &TextBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &TextBlock::isDense,
           "True if the block produces dense features, False otherwise.");

  py::class_<CategoricalBlock, Block, std::shared_ptr<CategoricalBlock>>(
      block_submodule, "Categorical",
      "A block that encodes categorical features (e.g. a numerical ID or an "
      "identification string).")
      .def(py::init<uint32_t, std::shared_ptr<CategoricalEncoding>>(),
           py::arg("col"), py::arg("encoding"),
           "Constructor.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the categorical feature to be encoded.\n"
           " * encoding: CategoricalEncoding - Categorical feature encoding "
           "model")
      .def(py::init<uint32_t, uint32_t>(), py::arg("col"), py::arg("dim"),
           "Constructor with default encoder.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the categorical feature to be encoded.\n"
           " * dim: Int - Dimension of the encoding")
      .def("feature_dim", &CategoricalBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &CategoricalBlock::isDense,
           "True if the block produces dense features, False otherwise.");

  py::class_<DenseArrayBlock, Block, std::shared_ptr<DenseArrayBlock>>(
      block_submodule, "DenseArray",
      "Parses a contiguous set of columns as a dense vector segment.")
      .def(py::init<uint32_t, uint32_t>(), py::arg("start_col"), py::arg("dim"),
           "Constructor")
      .def("feature_dim", &DenseArrayBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &DenseArrayBlock::isDense,
           "Returns true since this is a dense encoding.");

  py::class_<MockBlock, Block, std::shared_ptr<MockBlock>>(
      internal_dataset_submodule, "MockBlock",
      "Mock implementation of block abstract class for testing purposes.")
      .def(py::init<uint32_t, bool>(), py::arg("column"), py::arg("dense"),
           "Constructor")
      .def("feature_dim", &MockBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &MockBlock::isDense,
           "True if the block produces dense features, False otherwise.");

  py::class_<PyBlockBatchProcessor>(
      internal_dataset_submodule, "BatchProcessor",
      "Encodes input samples – each represented by a sequence of strings – "
      "as input and target BoltVectors according to the given blocks. "
      "It processes these sequences in batches.\n\n"
      "This is not consumer-facing.")
      .def(
          py::init<std::vector<std::shared_ptr<Block>>,
                   std::vector<std::shared_ptr<Block>>, uint32_t, size_t>(),
          py::arg("input_blocks"), py::arg("target_blocks"),
          py::arg("output_batch_size"), py::arg("est_num_elems") = 0,
          "Constructor\n\n"
          "Arguments:\n"
          " * input_blocks: List of Blocks - Blocks that encode input samples "
          "as input vectors.\n"
          " * target_blocks: List of Blocks - Blocks that encode input samples "
          "as target vectors.\n"
          " * output_batch_size: Int (positive) - Size of batches in the "
          "produced dataset.\n"
          " * est_num_elems: Int (positive) - Estimated number of samples. "
          "This "
          "speeds up the loading process by allowing the data loader to "
          "preallocate memory. If the actual number of samples turns out to be "
          "greater than the estimate, then the loader will automatically "
          "allocate more memory as needed.")
      .def("process_batch", &PyBlockBatchProcessor::processBatchPython,
           py::arg("row_batch"),
           "Consumes a batch of input samples and encodes them as vectors.\n\n"
           "Arguments:\n"
           " * row_batch: List of lists of strings - We expect to read tabular "
           "data "
           "where each row is a sample, and each sample has many columns. "
           "row_batch represents a batch of such samples.")
      .def("export_in_memory_dataset",
           &PyBlockBatchProcessor::exportInMemoryDataset,
           py::arg("shuffle") = false, py::arg("shuffle_seed") = std::rand(),
           "Produces a tuple of BoltDatasets for input and target "
           "vectors processed so far. This method can optionally produce a "
           "shuffled dataset.\n\n"
           "Arguments:\n"
           " * shuffle: Boolean (Optional) - The dataset will be shuffled if "
           "True.\n"
           " * shuffle_seed: Int (Optional) - The seed for the RNG for "
           "shuffling the "
           "dataset.");

  py::class_<StreamingGenericDatasetLoader>(dataset_submodule, "DataPipeline")
      .def(
          py::init<std::string, std::vector<std::shared_ptr<Block>>,
                   std::vector<std::shared_ptr<Block>>, uint32_t, bool, char>(),
          py::arg("filename"), py::arg("input_blocks"), py::arg("label_blocks"),
          py::arg("batch_size"), py::arg("has_header") = false,
          py::arg("delimiter") = ',')
      .def("next_batch", &StreamingGenericDatasetLoader::nextBatch)
      .def("load_in_memory", &StreamingGenericDatasetLoader::loadInMemory)
      .def("get_max_batch_size",
           &StreamingGenericDatasetLoader::getMaxBatchSize)
      .def("get_input_dim", &StreamingGenericDatasetLoader::getInputDim)
      .def("get_label_dim", &StreamingGenericDatasetLoader::getLabelDim);

  dataset_submodule.def("load_svm_dataset", &loadSVMDataset,
                        py::arg("filename"), py::arg("batch_size"));

  dataset_submodule.def("load_csv_dataset", &loadCSVDataset,
                        py::arg("filename"), py::arg("batch_size"),
                        py::arg("delimiter") = ",");

  dataset_submodule.def("make_sparse_vector", &BoltVector::makeSparseVector,
                        py::arg("indices"), py::arg("values"));

  dataset_submodule.def("make_dense_vector", &BoltVector::makeDenseVector,
                        py::arg("values"));

  // The no lint below is because clang tidy doesn't like instantiating an
  // object without a name and never using it.
  py::class_<ClickThroughDataset, ClickThroughDatasetPtr>(  // NOLINT
      dataset_submodule, "ClickThroughDataset");

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
      "Each line of the input file should follow this format:\n\n"
      "Returns a tuple containing a ClickthroughDataset to store the data "
      "itself, and a BoltDataset storing the labels.");

  py::class_<BoltDataset, BoltDatasetPtr>(dataset_submodule, "BoltDataset")
      .def("get",
           static_cast<bolt::BoltBatch& (BoltDataset::*)(uint32_t i)>(
               &BoltDataset::at),
           py::arg("i"), py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<bolt::BoltBatch& (BoltDataset::*)(uint32_t i)>(
               &BoltDataset::at),
           py::arg("i"), py::return_value_policy::reference);

  py::class_<bolt::BoltBatch>(dataset_submodule, "BoltBatch")
      .def("size", &bolt::BoltBatch::getBatchSize)
      .def("get",
           static_cast<BoltVector& (bolt::BoltBatch::*)(size_t i)>(
               &bolt::BoltBatch::operator[]),
           py::arg("i"), py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<BoltVector& (bolt::BoltBatch::*)(size_t i)>(
               &bolt::BoltBatch::operator[]),
           py::arg("i"), py::return_value_policy::reference);

  dataset_submodule.def(
      "load_bolt_svm_dataset", &loadBoltSvmDatasetWrapper, py::arg("filename"),
      py::arg("batch_size"), py::arg("softmax_for_multiclass") = true,
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
      " * batch_size: Int (positive) - Size of each batch in the dataset.\n"
      " * softmax_for_multiclass: Bool (default is true) - Multi-label samples "
      "must be processed slightly differently if softmax is being used in the "
      "output layer instead of sigmoid. When this flag is true the loader will "
      "process samples with multiple labels assuming that softmax and "
      "CategoricalCrossEntropy are being used for multi-label datasets. If the "
      "dataset is single label, then this argument has no effect.\n\n"
      "Returns a tuple containing a BoltDataset to store the data itself, and "
      "a BoltDataset storing the labels.");

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
      "line. Defaults to ','\n\n"
      "Returns a tuple containing a BoltDataset to store the data itself, and "
      "a BoltDataset storing the labels.");

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

  py::class_<InMemoryDataset<MaskedSentenceBatch>,
             std::shared_ptr<InMemoryDataset<MaskedSentenceBatch>>>(
      dataset_submodule, "MLMDataset")
      .def_static("load", &loadMLMDataset, py::arg("filename"),
                  py::arg("batch_size"), py::arg("pairgram_range"));

  internal_dataset_submodule.def(
      "dense_bolt_dataset_matches_dense_matrix",
      &denseBoltDatasetMatchesDenseMatrix, py::arg("dataset"),
      py::arg("matrix"),
      "Checks whether the given bolt dataset and dense 2d matrix "
      "have the same values. For testing purposes only.");

  internal_dataset_submodule.def(
      "dense_bolt_dataset_is_permutation_of_dense_matrix",
      &denseBoltDatasetIsPermutationOfDenseMatrix, py::arg("dataset"),
      py::arg("matrix"),
      "Checks whether the given bolt dataset represents a permutation of "
      "the rows of the given dense 2d matrix. Assumes that each row of "
      "the matrix is 1-dimensional; only has one element. For testing "
      "purposes only.");

  internal_dataset_submodule.def(
      "dense_bolt_datasets_are_equal", &denseBoltDatasetsAreEqual,
      py::arg("dataset1"), py::arg("dataset2"),
      "Checks whether the given bolt datasets have the same values. "
      "For testing purposes only.");
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
                                    uint32_t batch_size,
                                    bool softmax_for_multiclass) {
  auto res = loadBoltSvmDataset(filename, batch_size, softmax_for_multiclass);
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
        "For now, Numpy dense data must be 2D (each row is a dense data "
        "vector) or 1D (each element is treated as a row).");
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

BoltDatasetPtr sparseBoltDatasetFromNumpy(const NumpyArray<uint32_t>& indices,
                                          const NumpyArray<float>& values,
                                          const NumpyArray<uint32_t>& offsets,
                                          uint32_t batch_size) {
  uint64_t num_examples = static_cast<uint64_t>(offsets.shape(0) - 1);

  uint32_t* indices_raw_data = const_cast<uint32_t*>(indices.data());
  float* values_raw_data = const_cast<float*>(values.data());
  uint32_t* offsets_raw_data = const_cast<uint32_t*>(offsets.data());

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

BoltDatasetPtr categoricalLabelsFromNumpy(const NumpyArray<uint32_t>& labels,
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

std::unordered_map<uint32_t, uint32_t> parseSentenceToUnigrams(
    const std::string& sentence, uint32_t seed, uint32_t dimension) {
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

  return idx_to_val_map;
}

BoltVector parseSentenceToBoltVector(const std::string& sentence, uint32_t seed,
                                     uint32_t dimension) {
  std::unordered_map<uint32_t, uint32_t> idx_to_val_map =
      parseSentenceToUnigrams(sentence, seed, dimension);

  BoltVector vec(idx_to_val_map.size(), false, false);
  uint32_t i = 0;
  for (auto [index, value] : idx_to_val_map) {
    vec.active_neurons[i] = index;
    vec.activations[i] = value;
    i++;
  }

  return vec;
}

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>>
parseSentenceToSparseArray(const std::string& sentence, uint32_t seed,
                           uint32_t dimension) {
  std::unordered_map<uint32_t, uint32_t> idx_to_val_map =
      parseSentenceToUnigrams(sentence, seed, dimension);

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

bool denseBoltDatasetMatchesDenseMatrix(
    BoltDataset& dataset, std::vector<std::vector<float>>& matrix) {
  uint32_t batch_size = dataset.at(0).getBatchSize();
  for (uint32_t batch_idx = 0; batch_idx < dataset.numBatches(); batch_idx++) {
    auto& batch = dataset.at(batch_idx);
    for (uint32_t vec_idx = 0; vec_idx < batch.getBatchSize(); vec_idx++) {
      auto& vec = batch[vec_idx];
      uint32_t row = batch_idx * batch_size + vec_idx;
      for (uint32_t col = 0; col < vec.len; col++) {
        if (matrix[row][col] != vec.activations[col]) {
          return false;
        }
      }
    }
  }

  return true;
}

bool denseBoltDatasetIsPermutationOfDenseMatrix(
    BoltDataset& dataset, std::vector<std::vector<float>>& matrix) {
  // If one is a permutation of the other, they must have the same
  // number of rows / vectors.
  if (dataset.len() != matrix.size()) {
    return false;
  }

  // Keep track of values in the matrix
  std::unordered_map<float, uint32_t> expected_values;
  for (const auto& row : matrix) {
    assert(row.size() == 1);
    // Assume each row is 1-dimensional.
    expected_values[row.at(0)]++;
  }

  // Since each row only has one element, and we made sure that
  // the bolt dataset and the matrix have the same number of
  // vectors / rows, we now only need to make sure that
  // the bolt dataset contains the right number of occurrences
  // of each value in the matrix.
  std::unordered_map<float, uint32_t> actual_values;
  for (uint32_t batch_idx = 0; batch_idx < dataset.numBatches(); batch_idx++) {
    auto& batch = dataset[batch_idx];
    for (uint32_t vec_idx = 0; vec_idx < batch.getBatchSize(); vec_idx++) {
      actual_values[batch[vec_idx].activations[0]]++;
    }
  }

  for (const auto& [val, count] : actual_values) {
    if (count != expected_values[val]) {
      return false;
    }
  }

  return true;
}

bool denseBoltDatasetsAreEqual(BoltDataset& dataset1, BoltDataset& dataset2) {
  for (uint32_t batch_idx = 0; batch_idx < dataset1.numBatches(); batch_idx++) {
    auto& batch1 = dataset1[batch_idx];
    auto& batch2 = dataset2[batch_idx];
    for (uint32_t vec_idx = 0; vec_idx < batch1.getBatchSize(); vec_idx++) {
      auto& vec1 = batch1[vec_idx];
      auto& vec2 = batch2[vec_idx];
      for (uint32_t elem_idx = 0; elem_idx < vec1.len; elem_idx++) {
        if (vec1.activations[elem_idx] != vec2.activations[elem_idx]) {
          return false;
        }
      }
    }
  }

  return true;
}

py::tuple loadMLMDataset(const std::string& filename, uint32_t batch_size,
                         uint32_t pairgram_range) {
  auto data_loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

  auto batch_processor =
      std::make_shared<thirdai::dataset::MaskedSentenceBatchProcessor>(
          pairgram_range);

  auto dataset =
      std::make_shared<dataset::StreamingDataset<dataset::MaskedSentenceBatch>>(
          data_loader, batch_processor);

  auto [data, labels] = dataset->loadInMemory();

  return py::make_tuple(py::cast(data), py::cast(labels));
}

}  // namespace thirdai::dataset::python