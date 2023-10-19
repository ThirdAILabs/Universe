#include "DatasetPython.h"
#include "PyDataSource.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/DatasetLoaderWrappers.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/InMemoryDataset.h>
#include <dataset/src/NumpyDataset.h>
#include <dataset/src/VectorBuffer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/featurizers/llm/TextClassificationFeaturizer.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <dataset/src/mach/MachIndex.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <dataset/tests/MockBlock.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sys/types.h>
#include <utils/Random.h>
#include <chrono>
#include <limits>
#include <optional>
#include <type_traits>
#include <unordered_map>

// TODO(Geordie): Split into smaller files.
// I'm thinking one for each submodule of dataset_submodule.
// E.g. in DatasetBlockPython.cc we would have a function with this signature:
// void createBlockSubsubmodule(py::module_& dataset_submodule,
//                              py::module_& internal_dataset_submodule);

namespace thirdai::dataset::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

void createDatasetSubmodule(py::module_& module) {
  // Separate submodule for bindings that we don't want to expose to users.
  auto internal_dataset_submodule = module.def_submodule("dataset_internal");

  // Everything in this submodule is exposed to users.
  auto dataset_submodule = module.def_submodule("dataset");
  auto block_submodule = dataset_submodule.def_submodule("blocks");

  py::class_<BoltVector>(dataset_submodule, "BoltVector")
      .def("to_string", &BoltVector::toString)
      .def("__str__", &BoltVector::toString)
      .def("__repr__", &BoltVector::toString)
      .def("to_numpy", [](const BoltVector& vector) -> py::object {
        NumpyArray<float> activations_array(vector.len);
        std::copy(vector.activations, vector.activations + vector.len,
                  activations_array.mutable_data());

        if (vector.isDense()) {
          return py::object(std::move(activations_array));
        }

        NumpyArray<uint32_t> active_neurons_array(vector.len);
        std::copy(vector.active_neurons, vector.active_neurons + vector.len,
                  active_neurons_array.mutable_data());

        return py::make_tuple(active_neurons_array, activations_array);
      });

  py::class_<Explanation>(dataset_submodule, "Explanation",
                          R"pbdoc(
     Represents an input column that is responsible for a predicted
     outcome.
      )pbdoc")
      .def("__str__", &Explanation::toString)
      .def("__repr__", &Explanation::toString)
      // We don't expose column_number because it doesn't always make sense to
      // provide one column number; e.g. in tabular pairgram case.
      .def_readonly("percentage_significance",
                    &Explanation::percentage_significance,
                    R"pbdoc(
     The column's contribution to the predicted outcome. Can be 
     positive or negative depending on the relationship between 
     the responsible column and the prediction. For example, it
     can be negative if the responsible input column and the 
     prediction are negatively correlated.
      )pbdoc")
      .def_readonly("keyword", &Explanation::keyword,
                    R"pbdoc(
     A brief description of the value in this column.
      )pbdoc")
      .def_readonly("column_name", &Explanation::column_name,
                    R"pbdoc(
     Identifies the responsible input column.
      )pbdoc");

  py::class_<mach::MachIndex, mach::MachIndexPtr>(  // NOLINT
      dataset_submodule, "MachIndex")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<std::unordered_map<uint32_t, std::vector<uint32_t>>,
                    uint32_t, uint32_t>(),
           py::arg("entity_to_hashes"), py::arg("output_range"),
           py::arg("num_hashes"))
      .def(py::init<uint32_t, uint32_t>(), py::arg("output_range"),
           py::arg("num_hashes"))
      .def("get_entity_hashes", &mach::MachIndex::getHashes, py::arg("entity"))
      .def("get_hash_to_entities", &mach::MachIndex::getEntities,
           py::arg("hash"))
#endif
      .def("num_hashes", &mach::MachIndex::numHashes)
      .def("output_range", &mach::MachIndex::numBuckets)
      .def("save", &mach::MachIndex::save, py::arg("filename"))
      .def_static("load", &mach::MachIndex::load);

  py::class_<Block, std::shared_ptr<Block>>(
      internal_dataset_submodule, "Block",
      "Block abstract class.\n\n"
      "A block accepts an input sample in the form of a sequence of strings "
      "then encodes this sequence as a vector.")
      .def("feature_dim", &Block::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &Block::isDense,
           "True if the block produces dense features, False otherwise.");

  py::class_<TextTokenizer, TextTokenizerPtr>(  // NOLINT
      internal_dataset_submodule, "TextTokenizer");

  py::class_<CharKGramTokenizer, TextTokenizer,
             std::shared_ptr<CharKGramTokenizer>>(dataset_submodule,
                                                  "CharKGramTokenizer")
      .def(py::init<uint32_t>(), py::arg("k"));

  py::class_<NaiveSplitTokenizer, TextTokenizer,
             std::shared_ptr<NaiveSplitTokenizer>>(dataset_submodule,
                                                   "NaiveSplitTokenizer")
      .def(py::init<char>(), py::arg("delimiter"));

  py::class_<WordPunctTokenizer, TextTokenizer,
             std::shared_ptr<WordPunctTokenizer>>(dataset_submodule,
                                                  "WordPunctTokenizer")
      .def(py::init<>());

  py::class_<WordpieceTokenizer, TextTokenizer, WordpieceTokenizerPtr>(
      dataset_submodule, "WordpieceTokenizer")
      .def(py::init<std::string, bool>(), py::arg("vocab_file_path"),
           py::arg("lower_case") = true)
      .def("size", &WordpieceTokenizer::size)
      .def("unk_id", &WordpieceTokenizer::unkId)
      .def("mask_id", &WordpieceTokenizer::maskId)
      .def("tokenize", &WordpieceTokenizer::tokenize, py::arg("sequence"))
      .def("decode", &WordpieceTokenizer::decode, py::arg("piece_ids"))
      .def("id", &WordpieceTokenizer::id, py::arg("token"));

  py::class_<TextEncoder, TextEncoderPtr>(  // NOLINT
      dataset_submodule, "TextEncoder");

  py::class_<NGramEncoder, TextEncoder, std::shared_ptr<NGramEncoder>>(
      dataset_submodule, "NGramEncoder")
      .def(py::init<uint32_t>(), py::arg("n"));

  py::class_<PairGramEncoder, TextEncoder, std::shared_ptr<PairGramEncoder>>(
      dataset_submodule, "PairGramEncoder")
      .def(py::init<>());

  py::class_<TextBlock, Block, TextBlockPtr>(block_submodule, "TextBlock")
      .def(py::init<uint32_t, TextTokenizerPtr, TextEncoderPtr, bool,
                    uint32_t>(),
           py::arg("col"), py::arg("tokenizer") = NaiveSplitTokenizer::make(),
           py::arg("encoder") = NGramEncoder::make(1),
           py::arg("lowercase") = false,
           py::arg("dim") = token_encoding::DEFAULT_TEXT_ENCODING_DIM)
      .def("is_dense", &TextBlock::isDense)
      .def("feature_dim", &TextBlock::featureDim);

  py::class_<CategoricalBlock, Block, CategoricalBlockPtr>(
      internal_dataset_submodule, "AbstractCategoricalBlock",
      "A block that encodes categorical features (e.g. a numerical ID or "
      "an identification string).")
      .def("is_dense", &CategoricalBlock::isDense,
           "Returns false since categorical blocks always produce sparse "
           "features.")
      .def("feature_dim", &CategoricalBlock::featureDim,
           "The dimension of the vector encoding.");

  py::class_<NumericalCategoricalBlock, CategoricalBlock,
             NumericalCategoricalBlockPtr>(
      block_submodule, "NumericalId",
      "A block that encodes categories represented as numerical IDs.")
      .def(py::init<uint32_t, uint32_t, std::optional<char>>(), py::arg("col"),
           py::arg("n_classes"), py::arg("delimiter") = std::nullopt,
           "Constructor.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the categorical information to be encoded.\n"
           " * n_classes: Int - Number of unique categories.\n"
           " * delimiter: Char (Optional) - A character that separates "
           "different categories in the column. If not supplied, it is assumed "
           "that the column only contains a single class.")
      .def("feature_dim", &NumericalCategoricalBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &NumericalCategoricalBlock::isDense,
           "Returns false since text blocks always produce sparse features.");

  py::class_<DateBlock, Block, std::shared_ptr<DateBlock>>(
      block_submodule, "Date",
      "Encodes a date column given in YYYY-MM-DD format.")
      .def(py::init<uint32_t>(), py::arg("col"))
      .def("feature_dim", &DateBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &DateBlock::isDense,
           "Returns false since this is a sparse encoding.");

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

#if THIRDAI_EXPOSE_ALL
  py::class_<TabularColumn, std::shared_ptr<TabularColumn>>(
      dataset_submodule, "TabularColumn",
      "Column configuration for TabularHashFeatures block.")
      .def_static("Categorical", &TabularColumn::Categorical,
                  py::arg("column_identifier"))
      .def_static("Numeric", &TabularColumn::Numeric,
                  py::arg("column_identifier"), py::arg("range"),
                  py::arg("num_bins") = DEFAULT_NUM_BINS);

  py::class_<TabularHashFeatures, Block, std::shared_ptr<TabularHashFeatures>>(
      block_submodule, "TabularHashFeatures",
      "Given some metadata about a tabular dataset, assign unique "
      "categories to columns and compute either pairgramsor unigrams of the "
      "categories depending on the 'use_pairgrams' flag.")
      .def(py::init<std::vector<TabularColumn>, uint32_t, bool>(),
           py::arg("columns"), py::arg("output_range"),
           py::arg("use_pairgrams"))
      .def("feature_dim", &TabularHashFeatures::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &TabularHashFeatures::isDense,
           "Returns false since text blocks always produce sparse "
           "features.");

  py::class_<Featurizer, FeaturizerPtr>(dataset_submodule,  // NOLINT
                                        "Featurizer");

  py::class_<BlockList, BlockListPtr>(dataset_submodule, "BlockList")
      .def(py::init<std::vector<BlockPtr>, std::optional<uint32_t>>(),
           py::arg("block_lists"), py::arg("hash_range") = std::nullopt);

  py::class_<TabularFeaturizer, Featurizer, TabularFeaturizerPtr>(
      dataset_submodule, "TabularFeaturizer")
      .def(py::init<std::vector<BlockList>, bool, char, bool>(),
           py::arg("block_lists"), py::arg("has_header") = false,
           py::arg("delimiter") = ',', py::arg("parallel") = true);

  py::class_<TextGenerationFeaturizer, Featurizer,
             std::shared_ptr<TextGenerationFeaturizer>>(
      dataset_submodule, "TextGenerationFeaturizer")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, bool, bool>(),
           py::arg("lrc_len"), py::arg("irc_len"), py::arg("src_len"),
           py::arg("vocab_size"), py::arg("include_position") = false,
           py::arg("featurize_in_chunks") = true)
      .def("featurize_for_inference",
           &TextGenerationFeaturizer::featurizeInferenceSample,
           py::arg("prompt"), py::arg("context"))
      .def(bolt::python::getPickleFunction<TextGenerationFeaturizer>());

  py::class_<TextClassificationFeaturizer, Featurizer,
             TextClassificationFeaturizerPtr>(dataset_submodule,
                                              "TextClassificationFeaturizer")
      .def(py::init<const std::string&, const std::string&, uint32_t, uint32_t,
                    uint32_t, uint32_t, uint32_t, char, std::optional<char>,
                    bool, bool>(),
           py::arg("text_column"), py::arg("label_column"), py::arg("lrc_len"),
           py::arg("irc_len"), py::arg("src_len"), py::arg("vocab_size"),
           py::arg("n_labels"), py::arg("delimiter") = ',',
           py::arg("label_delimiter") = std::nullopt,
           py::arg("integer_labels") = false,
           py::arg("normalize_categories") = true);

#endif

  py::class_<DatasetShuffleConfig>(dataset_submodule, "ShuffleConfig")
      .def(py::init<size_t, uint32_t>(), py::arg("min_vecs_in_buffer") = 64000,
           py::arg("seed") = global_random::nextSeed());

  py::class_<DatasetLoader, DatasetLoaderPtr>(dataset_submodule,
                                              "DatasetLoader")
      .def(py::init<std::shared_ptr<dataset::DataSource>,
                    dataset::FeaturizerPtr, bool, DatasetShuffleConfig>(),
           py::arg("data_source"), py::arg("featurizer"), py::arg("shuffle"),
           py::arg("shuffle_config") = DatasetShuffleConfig())
      .def("get_input_dim", &DatasetLoader::getInputDim)
      .def("get_label_dim", &DatasetLoader::getLabelDim)
      .def("load_all", &DatasetLoader::loadAll, py::arg("batch_size"),
           py::arg("verbose") = true)
      .def("load_some", &dataset::DatasetLoader::loadSome,
           py::arg("batch_size"), py::arg("num_batches"),
           py::arg("verbose") = true)
      .def("restart", &dataset::DatasetLoader::restart);

  py::class_<DataSource, PyDataSource, DataSourcePtr>(dataset_submodule,
                                                      "DataSource")
      .def(py::init<>())
      .def("next_batch", &DataSource::nextBatch)
      .def("next_line", &DataSource::nextLine)
      .def("resource_name", &DataSource::resourceName)
      .def("restart", &DataSource::restart);

  py::class_<cold_start::ColdStartDataSource, dataset::DataSource,
             cold_start::ColdStartDataSourcePtr>
      ColdStartDataSource(dataset_submodule, "ColdStartDataSource");

  py::class_<FileDataSource, DataSource, std::shared_ptr<FileDataSource>>(
      dataset_submodule, "FileDataSource")
      .def(py::init<const std::string&>(), py::arg("filename"));

  dataset_submodule.def("make_sparse_vector",
                        py::overload_cast<const std::vector<uint32_t>&,
                                          const std::vector<float>&>(
                            &BoltVector::makeSparseVector),
                        py::arg("indices"), py::arg("values"));

  dataset_submodule.def("make_dense_vector", &BoltVector::makeDenseVector,
                        py::arg("values"));

  dataset_submodule.def(
      "load_click_through_dataset",
      &ClickThroughDatasetLoader::loadDatasetFromFile, py::arg("filename"),
      py::arg("batch_size"), py::arg("max_num_numerical_features"),
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
           static_cast<BoltBatch& (BoltDataset::*)(uint64_t i)>(
               &BoltDataset::at),
           py::arg("i"), py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<BoltBatch& (BoltDataset::*)(uint64_t i)>(
               &BoltDataset::at),
           py::arg("i"), py::return_value_policy::reference)
      .def("__len__", &BoltDataset::numBatches)
      .def("save", &BoltDataset::save, py::arg("filename"))
      .def_static("load", &BoltDataset::load, py::arg("filename"))
      .def(py::init([](const py::iterable& iterable) {
             using Batches = std::vector<BoltBatch>;
             auto batches = iterable.cast<Batches>();
             std::shared_ptr<BoltDataset> dataset =
                 std::make_shared<InMemoryDataset>(std::move(batches));
             return dataset;
           }),
           py::arg("batches"), R"pbdoc(
            Construct a BoltDataset from an iterable of BoltBatches. Makes
            copies in the process which can potentially be costly, use judiciously.
            
            Args: 
                batches (Iterable[BoltBatch]): Batches

            Returns:
                BoltDataset: The constructed dataset.
           )pbdoc");

  py::class_<numpy::NumpyInMemoryDataset,  // NOLINT
             std::shared_ptr<numpy::NumpyInMemoryDataset>, BoltDataset>(
      dataset_submodule, "NumpyInMemoryDataset");

  // TODO(Any): Add __iter__ method so we can do foreach loops in python and c++
  // TODO(Any): This segfaults if the user passes in an index that is too large
  py::class_<BoltBatch>(dataset_submodule, "BoltBatch")
      .def("batch_size", &BoltBatch::getBatchSize)
      // We need to explicitly static cast these methods because there are
      // multiple candidate "[]" methods (one const and one not const)
      .def("get",
           static_cast<BoltVector& (BoltBatch::*)(size_t i)>(
               &BoltBatch::operator[]),
           py::arg("i"), py::return_value_policy::reference)
      .def("__getitem__",
           static_cast<BoltVector& (BoltBatch::*)(size_t i)>(
               &BoltBatch::operator[]),
           py::arg("i"), py::return_value_policy::reference)
      .def("__len__", &BoltBatch::getBatchSize)
      .def(py::init([](const py::iterable& iterable) {
             using Vectors = std::vector<BoltVector>;
             auto vectors = iterable.cast<Vectors>();
             BoltBatch batch(std::move(vectors));
             return batch;
           }),
           py::arg("vectors"), R"pbdoc(
            Construct a BoltBatch from an iterable of BoltVectors. Makes copies
            in the process which can be costly, use judiciously.

            Args: 
                vectors (Iterable[BoltVector]): BoltVectors constituting the Batch.

            Returns:
                BoltBatch: The constructed Batch.
           )pbdoc");

  dataset_submodule.def(
      "load_bolt_svm_dataset", SvmDatasetLoader::loadDatasetFromFile,
      py::arg("filename"), py::arg("batch_size"),
      py::arg("softmax_for_multiclass") = true,
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
      "load_bolt_svm_dataset", SvmDatasetLoader::loadDataset,
      py::arg("data_source"), py::arg("batch_size"),
      py::arg("softmax_for_multiclass") = true,
      "The same as the other implementation of this method, but takes in a "
      "custom data source instead of a file name.");

  dataset_submodule.def("from_numpy", &numpy::numpyToBoltVectorDataset,
                        py::arg("data"), py::arg("batch_size") = std::nullopt);

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
    for (const auto& vec : batch) {
      actual_values[vec.activations[0]]++;
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
