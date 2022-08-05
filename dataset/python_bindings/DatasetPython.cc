#include "DatasetPython.h"
#include <dataset/src/DatasetLoaders.h>
#include <dataset/src/NumpyDataset.h>
#include <dataset/src/ShuffleBatchBuffer.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/MaskedSentenceBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/Text.h>
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

  py::class_<DatasetShuffleConfig>(dataset_submodule, "ShuffleBufferConfig")
      .def(py::init<size_t, uint32_t>(), py::arg("n_batches") = 1000,
           py::arg("seed") = time(NULL));

  py::class_<StreamingGenericDatasetLoader>(dataset_submodule, "DataPipeline")
      .def(py::init<std::string, std::vector<std::shared_ptr<Block>>,
                    std::vector<std::shared_ptr<Block>>, uint32_t, bool,
                    DatasetShuffleConfig, bool, char>(),
           py::arg("filename"), py::arg("input_blocks"),
           py::arg("label_blocks"), py::arg("batch_size"),
           py::arg("shuffle") = false,
           py::arg("config") = DatasetShuffleConfig(),
           py::arg("has_header") = false, py::arg("delimiter") = ',')
      .def("next_batch", &StreamingGenericDatasetLoader::nextBatchTuple)
      .def("load_in_memory", &StreamingGenericDatasetLoader::loadInMemory)
      .def("get_max_batch_size",
           &StreamingGenericDatasetLoader::getMaxBatchSize)
      .def("get_input_dim", &StreamingGenericDatasetLoader::getInputDim)
      .def("get_label_dim", &StreamingGenericDatasetLoader::getLabelDim);

  dataset_submodule.def("make_sparse_vector", &BoltVector::makeSparseVector,
                        py::arg("indices"), py::arg("values"));

  dataset_submodule.def("make_dense_vector", &BoltVector::makeDenseVector,
                        py::arg("values"));

  dataset_submodule.def(
      "load_click_through_dataset", &ClickThroughDatasetLoader::loadDataset,
      py::arg("filename"), py::arg("batch_size"),
      py::arg("max_num_numerical_features"),
      py::arg("max_categorical_features"), py::arg("delimiter") = '\t',
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
      " * max_categorical_features: Int (positive) - Maximum number of "
      "expected categorical features in each dataset.\n"
      "Returns a tuple containing a BoltDataset, BoltTokenDataset to store the "
      "dense and categorical features, and a BoltDataset storing the labels.");

  py::class_<BoltDataset, BoltDatasetPtr>(dataset_submodule, "BoltDataset")
      // We need to explicitly static cast these methods because there are
      // multiple candidate "at" methods (one const and one not const)
      .def("get",
           static_cast<bolt::BoltBatch& (BoltDataset::*)(uint64_t i)>(
               &BoltDataset::at),
           py::arg("i"), py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<bolt::BoltBatch& (BoltDataset::*)(uint64_t i)>(
               &BoltDataset::at),
           py::arg("i"), py::return_value_policy::reference)
      .def("__len__", &BoltDataset::numBatches);

  py::class_<BoltTokenDataset, BoltTokenDatasetPtr>(  // NOLINT
      dataset_submodule, "BoltTokenDataset");

  py::class_<numpy::WrappedNumpyVectors,  // NOLINT
             std::shared_ptr<numpy::WrappedNumpyVectors>, BoltDataset>(
      dataset_submodule, "WrappedNumpyVectors");
  py::class_<numpy::WrappedNumpyTokens,  // NOLINT
             std::shared_ptr<numpy::WrappedNumpyTokens>, BoltTokenDataset>(
      dataset_submodule, "WrappedNumpyTokens");

  // TODO(josh): Add __iter__ method so we can do foreach loops in pthon and c++
  // TODO(josh): This segfaults if the user passes in an index that is too large
  py::class_<bolt::BoltBatch>(dataset_submodule, "BoltBatch")
      .def("batch_size", &bolt::BoltBatch::getBatchSize)
      // We need to explicitly static cast these methods because there are
      // multiple candidate "[]" methods (one const and one not const)
      .def("get",
           static_cast<BoltVector& (bolt::BoltBatch::*)(size_t i)>(
               &bolt::BoltBatch::operator[]),
           py::arg("i"), py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<BoltVector& (bolt::BoltBatch::*)(size_t i)>(
               &bolt::BoltBatch::operator[]),
           py::arg("i"), py::return_value_policy::reference)
      .def("__len__", &bolt::BoltBatch::getBatchSize);

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

  dataset_submodule.def("from_numpy", &numpy::numpyToBoltVectorDataset,
                        py::arg("data"), py::arg("batch_size") = std::nullopt);

  dataset_submodule.def("tokens_from_numpy", &numpy::numpyToBoltTokenDataset,
                        py::arg("data"), py::arg("batch_size") = std::nullopt);

  dataset_submodule.def(
      "bolt_tokenizer", &parseSentenceToUnigramsPython, py::arg("sentence"),
      py::arg("dimension") = 100000,
      "Utility that turns a sentence into a sequence of token embeddings. To "
      "be used for text classification tasks.\n"
      "Arguments:\n"
      " * sentence: String - Sentence to be tokenized.\n"
      " * dimensions: Int (positive) - (Optional) The dimension of each token "
      "embedding. "
      "Defaults to 100,000.");

  py::class_<MLMDatasetLoader>(dataset_submodule, "MLMDatasetLoader")
      .def(py::init<uint32_t>(), py::arg("pairgram_range"))
      .def("load", &MLMDatasetLoader::load, py::arg("filename"),
           py::arg("batch_size"));

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

py::tuple loadBoltSvmDatasetWrapper(const std::string& filename,
                                    uint32_t batch_size,
                                    bool softmax_for_multiclass) {
  auto [data, labels] = SvmDatasetLoader::loadDataset(filename, batch_size,
                                                      softmax_for_multiclass);
  return py::make_tuple(std::move(data), std::move(labels));
}

std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>>
parseSentenceToUnigramsPython(const std::string& sentence, uint32_t dimension) {
  std::vector<uint32_t> unigrams =
      TextEncodingUtils::computeRawUnigramsWithRange(sentence, dimension);

  std::vector<uint32_t> indices;
  std::vector<uint32_t> values;
  TextEncodingUtils::sumRepeatedIndices(unigrams, /* base_value= */ 1.0,
                                        [&](uint32_t index, float value) {
                                          indices.push_back(index);
                                          values.push_back(value);
                                        });

  auto result = py::array_t<uint32_t>(indices.size());
  py::buffer_info indx_buf = result.request();
  uint32_t* indx_ptr = static_cast<uint32_t*>(indx_buf.ptr);

  auto result_2 = py::array_t<uint32_t>(values.size());
  py::buffer_info val_buf = result_2.request();
  uint32_t* val_ptr = static_cast<uint32_t*>(val_buf.ptr);

  assert(indices.size() == values.size());
  for (uint32_t i = 0; i < indices.size(); i++) {
    indx_ptr[i] = indices[i];
    val_ptr[i] = values[i];
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

}  // namespace thirdai::dataset::python