#include "DatasetPython.h"
#include "PyDataLoader.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/DatasetLoaders.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/InMemoryDataset.h>
#include <dataset/src/NumpyDataset.h>
#include <dataset/src/ShuffleBatchBuffer.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/Vocabulary.h>
#include <dataset/src/batch_processors/MaskedSentenceBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/TabularPairGram.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/data_pipeline/FeaturizationPipeline.h>
#include <dataset/src/data_pipeline/Transformation.h>
#include <dataset/src/data_pipeline/columns/NumpyColumns.h>
#include <dataset/src/data_pipeline/transformations/Binning.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/tests/MockBlock.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <sys/types.h>
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
      .def_readonly("column_number", &Explanation::column_number,
                    R"pbdoc(
     Identifies the responsible input column.
      )pbdoc")
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

  py::class_<Block, std::shared_ptr<Block>>(
      internal_dataset_submodule, "Block",
      "Block abstract class.\n\n"
      "A block accepts an input sample in the form of a sequence of strings "
      "then encodes this sequence as a vector.")
      .def("feature_dim", &Block::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &Block::isDense,
           "True if the block produces dense features, False otherwise.");

  py::class_<TextBlock, Block, TextBlockPtr>(
      internal_dataset_submodule, "AbstractTextBlock",
      "Abstract block for processing text (e.g. sentences / paragraphs).")
      .def("is_dense", &TextBlock::isDense,
           "Returns false since text blocks always produce sparse features.")
      .def("feature_dim", &TextBlock::featureDim,
           "The dimension of the vector encoding.");

  py::class_<PairGramTextBlock, TextBlock, PairGramTextBlockPtr>(
      block_submodule, "TextPairGram",
      "A block that encodes text as a weighted set of ordered pairs of "
      "space-separated words.")
      .def(py::init<uint32_t, uint32_t>(), py::arg("col"),
           py::arg("dim") = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
           "Constructor.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the text to be encoded.\n"
           " * dim: Int - Dimension of the encoding")
      .def("feature_dim", &PairGramTextBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &PairGramTextBlock::isDense,
           "Returns false since text blocks always produce sparse features.");

  py::class_<UniGramTextBlock, TextBlock, UniGramTextBlockPtr>(
      block_submodule, "TextUniGram",
      "A block that encodes text as a weighted set of space-separated words.")
      .def(py::init<uint32_t, uint32_t>(), py::arg("col"),
           py::arg("dim") = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
           "Constructor.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the text to be encoded.\n"
           " * dim: Int - Dimension of the encoding")
      .def("feature_dim", &UniGramTextBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &UniGramTextBlock::isDense,
           "Returns false since text blocks always produce sparse "
           "features.");

  py::class_<CharKGramTextBlock, TextBlock, CharKGramTextBlockPtr>(
      block_submodule, "TextCharKGram",
      "A block that encodes text as a weighted set of character trigrams.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("col"),
           py::arg("k"),
           py::arg("dim") = TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
           "Constructor.\n\n"
           "Arguments:\n"
           " * col: Int - Column number of the input row containing "
           "the text to be encoded.\n"
           " * k: Int - Number of characters in each character k-gram token.\n"
           " * dim: Int - Dimension of the encoding")
      .def("feature_dim", &CharKGramTextBlock::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &CharKGramTextBlock::isDense,
           "Returns false since text blocks always produce sparse features.");

  py::class_<CategoricalBlock, Block, CategoricalBlockPtr>(
      internal_dataset_submodule, "AbstractCategoricalBlock",
      "A block that encodes categorical features (e.g. a numerical ID or an "
      "identification string).")
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
  py::enum_<TabularDataType>(dataset_submodule, "TabularDataType")
      .value("Label", TabularDataType::Label)
      .value("Categorical", TabularDataType::Categorical)
      .value("Numeric", TabularDataType::Numeric);

  py::class_<TabularMetadata, std::shared_ptr<TabularMetadata>>(
      dataset_submodule, "TabularMetadata", "Metadata for a tabular dataset.")
      .def(py::init(
               [](std::vector<TabularDataType> column_dtypes,
                  std::unordered_map<uint32_t, std::pair<double, double>>
                      col_min_maxes,
                  std::unordered_map<std::string, uint32_t> class_name_to_id,
                  std::vector<std::string> column_names = {},
                  std::optional<std::unordered_map<uint32_t, uint32_t>>
                      col_to_num_bins = std::nullopt) {
                 return std::make_shared<TabularMetadata>(
                     column_dtypes, col_min_maxes,
                     ThreadSafeVocabulary::make(std::move(class_name_to_id),
                                                /* fixed = */ true),
                     column_names, col_to_num_bins);
               }),
           py::arg("column_dtypes"), py::arg("col_min_maxes"),
           py::arg("class_name_to_id") =
               std::unordered_map<std::string, uint32_t>(),
           py::arg("column_names") = std::vector<std::string>(),
           py::arg("col_to_num_bins") = std::nullopt);

  py::class_<TabularPairGram, Block, std::shared_ptr<TabularPairGram>>(
      block_submodule, "TabularPairGram",
      "Given some metadata about a tabular dataset, assign unique "
      "categories "
      "to columns and compute pairgrams of the categories.")
      .def(py::init<std::shared_ptr<TabularMetadata>, uint32_t>(),
           py::arg("metadata"), py::arg("output_range"))
      .def("feature_dim", &TabularPairGram::featureDim,
           "Returns the dimension of the vector encoding.")
      .def("is_dense", &TabularPairGram::isDense,
           "Returns false since text blocks always produce sparse "
           "features.");
#endif

  py::class_<DataLoader, PyDataLoader, std::shared_ptr<DataLoader>>(
      dataset_submodule, "DataLoader")
      .def(py::init<uint32_t>(), py::arg("target_batch_size"))
      .def("next_batch", &DataLoader::nextBatch)
      .def("next_line", &DataLoader::nextLine)
      .def("resource_name", &DataLoader::resourceName)
      .def("restart", &DataLoader::restart);

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
      .def_static("load", &BoltDataset::load, py::arg("filename"));

  py::class_<numpy::WrappedNumpyVectors,  // NOLINT
             std::shared_ptr<numpy::WrappedNumpyVectors>, BoltDataset>(
      dataset_submodule, "WrappedNumpyVectors");

  // TODO(josh): Add __iter__ method so we can do foreach loops in pthon and c++
  // TODO(josh): This segfaults if the user passes in an index that is too large
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
      .def("__len__", &BoltBatch::getBatchSize);

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
      py::arg("data_loader"), py::arg("softmax_for_multiclass") = true,
      "The same as the other implementation of this method, but takes in a "
      "custom data loader instead of a file name.");

  dataset_submodule.def("from_numpy", &numpy::numpyToBoltVectorDataset,
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
      .def(py::init<std::shared_ptr<Vocabulary>, uint32_t>(),
           py::arg("vocabulary"), py::arg("pairgram_range"))
      .def(py::init<std::shared_ptr<Vocabulary>, uint32_t, float>(),
           py::arg("vocabulary"), py::arg("pairgram_range"),
           py::arg("masked_tokens_percentage"))
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

  py::class_<Vocabulary, std::shared_ptr<Vocabulary>>(dataset_submodule,
                                                      "Vocabulary")
      .def("size", &Vocabulary::size)
      .def("unk_id", &Vocabulary::unkId)
      .def("mask_id", &Vocabulary::maskId)
      .def("encode", &Vocabulary::encode, py::arg("sequence"))
      .def("decode", &Vocabulary::decode, py::arg("piece_ids"))
      .def("id", &Vocabulary::id, py::arg("token"));

  py::class_<FixedVocabulary, Vocabulary, std::shared_ptr<FixedVocabulary>>(
      dataset_submodule, "FixedVocabulary")
      .def_static("make", &FixedVocabulary::make, py::arg("vocab_file_path"));

  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<Column, ColumnPtr>(columns_submodule, "Column");  // NOLINT

  py::class_<NumpyValueColumn<uint32_t>, Column,
             std::shared_ptr<NumpyValueColumn<uint32_t>>>(
      columns_submodule, "NumpySparseValueColumn")
      .def(py::init<const NumpyArray<uint32_t>&, uint32_t>(), py::arg("array"),
           py::arg("dim"));

  py::class_<NumpyValueColumn<float>, Column,
             std::shared_ptr<NumpyValueColumn<float>>>(columns_submodule,
                                                       "NumpyDenseValueColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"));

  py::class_<NumpyArrayColumn<uint32_t>, Column,
             std::shared_ptr<NumpyArrayColumn<uint32_t>>>(
      columns_submodule, "NumpySparseArrayColumn")
      .def(py::init<const NumpyArray<uint32_t>&, uint32_t>(), py::arg("array"),
           py::arg("dim"));

  py::class_<NumpyArrayColumn<float>, Column,
             std::shared_ptr<NumpyArrayColumn<float>>>(columns_submodule,
                                                       "NumpyDenseArrayColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"));

  auto transformations_submodule =
      dataset_submodule.def_submodule("transformations");

  py::class_<Transformation, std::shared_ptr<Transformation>>(  // NOLINT
      transformations_submodule, "Transformation");

  py::class_<BinningTransformation, Transformation,
             std::shared_ptr<BinningTransformation>>(transformations_submodule,
                                                     "Binning")
      .def(py::init<std::string, std::string, float, float, uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("inclusive_min"), py::arg("exclusive_max"),
           py::arg("num_bins"));

  py::class_<ColumnMap>(dataset_submodule, "ColumnMap")
      .def(py::init<std::unordered_map<std::string, ColumnPtr>>(),
           py::arg("columns"))
      .def("convert_to_dataset", &ColumnMap::convertToDataset,
           py::arg("columns"), py::arg("batch_size"))
      .def("__getitem__", &ColumnMap::getColumn);

  py::class_<FeaturizationPipeline>(dataset_submodule, "FeaturizationPipeline")
      .def(py::init<std::vector<TransformationPtr>>(),
           py::arg("transformations"))
      .def("featurize", &FeaturizationPipeline::featurize, py::arg("columns"));
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
